# =============================================
# SPLIT DATASET - Run this first!
# =============================================
import os, shutil, random

SOURCE = 'dataset/PlantVillage'
TRAIN  = 'dataset/train'
VAL    = 'dataset/val'
SPLIT  = 0.8

print("Starting dataset split...")

for class_name in os.listdir(SOURCE):
    class_path = os.path.join(SOURCE, class_name)
    if not os.path.isdir(class_path): continue
    if class_name == 'PlantVillage': continue  # skip nested folder

    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if len(images) == 0: continue

    random.shuffle(images)
    split_idx = int(len(images) * SPLIT)

    os.makedirs(os.path.join(TRAIN, class_name), exist_ok=True)
    os.makedirs(os.path.join(VAL,   class_name), exist_ok=True)

    for img in images[:split_idx]:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(TRAIN, class_name, img))
    for img in images[split_idx:]:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(VAL, class_name, img))

    print(f"✅ {class_name}: {split_idx} train, {len(images)-split_idx} val")

print("\nDataset split complete! Ready to train.")
