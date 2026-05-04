import argparse
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

PLANT_VILLAGE_DIR = "Dataset/PlantVillage"
FARM_DATASET_DIR = "Dataset/ML DATASET"
YOLO_DATASET_DIR = "Dataset/ML DATASET.v2i.yolov8/train"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
SEED = 42

ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def is_image_file(filename):
    return filename.lower().endswith(ALLOWED_EXTENSIONS)


def collect_plantvillage_binary():
    dataset = []
    if not os.path.isdir(PLANT_VILLAGE_DIR):
        return dataset

    for root, _, files in os.walk(PLANT_VILLAGE_DIR):
        for filename in files:
            if not is_image_file(filename):
                continue
            filepath = os.path.join(root, filename)
            class_folder = os.path.basename(root).lower()
            label = 'healthy' if 'healthy' in class_folder else 'diseased'
            dataset.append((filepath, label))

    return dataset


def collect_farm_binary():
    dataset = []
    if not os.path.isdir(FARM_DATASET_DIR):
        return dataset

    for class_name in os.listdir(FARM_DATASET_DIR):
        folder_path = os.path.join(FARM_DATASET_DIR, class_name)
        if not os.path.isdir(folder_path):
            continue

        lower_name = class_name.lower()
        if 'out' in lower_name:
            label = 'diseased'
        elif 'grid' in lower_name:
            label = 'healthy'
        else:
            continue

        for filename in os.listdir(folder_path):
            if not is_image_file(filename):
                continue
            dataset.append((os.path.join(folder_path, filename), label))

    return dataset


def collect_yolo_binary():
    dataset = []
    image_dir = os.path.join(YOLO_DATASET_DIR, 'images')
    label_dir = os.path.join(YOLO_DATASET_DIR, 'labels')

    if not os.path.isdir(image_dir):
        return dataset

    for filename in os.listdir(image_dir):
        if not is_image_file(filename):
            continue

        filepath = os.path.join(image_dir, filename)
        base_name, _ = os.path.splitext(filename)
        label_file = os.path.join(label_dir, base_name + '.txt')
        label = 'healthy'

        if os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]

            if any(line.split()[0] == '0' for line in lines):
                label = 'diseased'
            elif any(line.split()[0] == '1' for line in lines):
                label = 'healthy'

        dataset.append((filepath, label))

    return dataset


def build_dataset_items():
    items = []
    items.extend(collect_plantvillage_binary())
    items.extend(collect_farm_binary())
    items.extend(collect_yolo_binary())

    random.seed(SEED)
    random.shuffle(items)
    return items


def split_items(items, split=0.8):
    grouped = {'healthy': [], 'diseased': []}
    for filepath, label in items:
        grouped[label].append(filepath)

    train, val = [], []
    for label, paths in grouped.items():
        random.shuffle(paths)
        split_index = int(len(paths) * split)
        train.extend([(p, label) for p in paths[:split_index]])
        val.extend([(p, label) for p in paths[split_index:]])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=2)
    total = labels.shape[0]
    weights = {}
    for i in range(2):
        weights[i] = float(total) / (2.0 * counts[i]) if counts[i] > 0 else 1.0
    return weights


def create_tf_dataset(items, batch_size, training=True):
    paths = [path for path, _ in items]
    labels = np.array([0 if label == 'healthy' else 1 for _, label in items], dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def parse_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)
        image = preprocess_input(image)
        return image, label

    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(buffer_size=1024, seed=SEED)
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def augment_image(image, label):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomContrast(0.1)
    ])
    image = augmentation(image)
    return image, label


def build_model():
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def fine_tune_model(model, train_dataset, val_dataset, initial_epochs):
    model.trainable = True
    fine_tune_at = 100

    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=initial_epochs + 10,
        callbacks=[
            ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        ]
    )
    return history


def main():
    parser = argparse.ArgumentParser(description='Train MobileNetV2 diseased/healthy model from PlantVillage, farm photos, and YOLO labels.')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of initial training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    items = build_dataset_items()
    if len(items) == 0:
        raise RuntimeError('No training images found. Please check your dataset folders.')

    train_items, val_items = split_items(items)
    print(f'✅ Total images: {len(items)}')
    print(f'✅ Train images: {len(train_items)}, Validation images: {len(val_items)}')

    train_dataset = create_tf_dataset(train_items, args.batch_size, training=True)
    val_dataset = create_tf_dataset(val_items, args.batch_size, training=False)
    y_train = np.array([0 if label == 'healthy' else 1 for _, label in train_items], dtype=np.int32)
    class_weights = compute_class_weights(y_train)
    print(f'✅ Class weights: {class_weights}')

    model = build_model()

    os.makedirs('model', exist_ok=True)
    callbacks = [
        ModelCheckpoint('model/plant_model_best.h5', save_best_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
    ]

    print('🚀 Initial training started...')
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print('🚀 Fine-tuning last MobileNetV2 layers...')
    fine_tune_model(model, train_dataset, val_dataset, args.epochs)

    os.makedirs('model', exist_ok=True)
    model.save('model/plant_model.h5')
    print('✅ Training Done! Model saved to model/plant_model.h5')


if __name__ == '__main__':
    main()
