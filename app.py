import sys
import os
import subprocess

# --- AUTO-ACTIVATE VIRTUAL ENVIRONMENT ---
# If we are not running inside the venv, automatically restart using the venv python
if hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix:
    pass # Already in venv
else:
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'Scripts', 'python.exe')
    if os.path.exists(venv_python):
        print("--> Automatically switching to virtual environment...")
        sys.exit(subprocess.call([venv_python] + sys.argv))
# -----------------------------------------

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
from PIL import Image
from PIL import ImageOps
import io
from werkzeug.utils import secure_filename

# Fix for UnicodeEncodeError with emojis on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_PATH = "model/plant_model.h5"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model and class names
model = None
class_names = []

DISEASE_SUGGESTIONS = {
    'healthy': 'Your plant appears healthy. Continue regular watering, balanced nutrition, and good air circulation.',
    'diseased': 'Plant appears diseased. Remove affected leaves, improve air circulation, and consider a broad-spectrum fungicide or pesticide depending on the symptoms.',
    'unknown': 'Unable to determine disease with high confidence. Try another image with better lighting and a clear view of the leaf.',
    'Apple___Apple_scab': 'Apple scab: remove infected leaves and treat with sulfur sprays or an appropriate fungicide.',
    'Apple___Black_rot': 'Apple black rot: prune infected areas and use copper-based fungicides.',
    'Apple___Cedar_apple_rust': 'Cedar apple rust: remove nearby junipers and spray with a fungicide.',
    'Cherry_(including_sour)___Powdery_mildew': 'Powdery mildew: apply sulfur or potassium bicarbonate sprays and improve airflow.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Gray leaf spot: apply fungicide and avoid overhead irrigation.',
    'Corn_(maize)___Common_rust_': 'Common rust: use rust-resistant corn varieties and fungicide if needed.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Northern leaf blight: remove debris and apply protective fungicides.',
    'Grape___Black_rot': 'Black rot: prune infected canes and use fungicides during the season.',
    'Grape___Esca_(Black_Measles)': 'Esca: remove infected wood and improve canopy ventilation.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Leaf blight: apply fungicides and avoid wet foliage at night.',
    'Orange___Haunglongbing_(Citrus_greening)': 'Citrus greening: remove infected trees and control psyllids with insecticides.',
    'Peach___Bacterial_spot': 'Bacterial spot: use copper sprays and remove infected fruit/foliage.',
    'Pepper,_bell___Bacterial_spot': 'Bacterial spot: avoid overhead watering and use copper-based sprays.',
    'Potato___Early_blight': 'Early blight: remove infected leaves and use fungicide sprays.',
    'Potato___Late_blight': 'Late blight: avoid wet foliage and apply a late blight fungicide.',
    'Strawberry___Leaf_scorch': 'Leaf scorch: improve drainage and reduce overhead watering.',
    'Tomato___Bacterial_spot': 'Bacterial spot: use copper sprays and avoid splashing soil on leaves.',
    'Tomato___Early_blight': 'Early blight: remove infected leaves and use protective fungicides.',
    'Tomato___Late_blight': 'Late blight: apply appropriate fungicides and remove infected plants.',
    'Tomato___Leaf_Mold': 'Leaf mold: increase air circulation and apply fungicides.',
    'Tomato___Septoria_leaf_spot': 'Septoria leaf spot: remove lower leaves and use fungicides.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spider mites: spray with water or miticide and keep humidity higher.',
    'Tomato___Target_Spot': 'Target spot: remove debris and consider fungicide applications.',
    'Tomato___Tomato_mosaic_virus': 'Tomato mosaic virus: remove infected plants and wash hands/tools between plants.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato yellow leaf curl virus: control whiteflies and remove infected plants.'
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model"""
    global model, class_names
    try:
        # Enable memory growth for GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Memory growth setting failed: {e}")

        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")

        output_shape = model.output_shape
        if isinstance(output_shape, tuple) and len(output_shape) >= 2:
            num_classes = output_shape[-1]
        else:
            num_classes = 1

        if num_classes == 2:
            class_names = ['healthy', 'diseased']
        else:
            class_names = [
                'Apple___Apple_scab',
                'Apple___Black_rot',
                'Apple___Cedar_apple_rust',
                'Apple___healthy',
                'Blueberry___healthy',
                'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy',
                'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)',
                'Peach___Bacterial_spot',
                'Peach___healthy',
                'Pepper,_bell___Bacterial_spot',
                'Pepper,_bell___healthy',
                'Potato___Early_blight',
                'Potato___Late_blight',
                'Potato___healthy',
                'Raspberry___healthy',
                'Soybean___healthy',
                'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch',
                'Strawberry___healthy',
                'Tomato___Bacterial_spot',
                'Tomato___Early_blight',
                'Tomato___Late_blight',
                'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot',
                'Tomato___Tomato_mosaic_virus',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___healthy'
            ]

        print(f"✅ Loaded {len(class_names)} class names")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def correct_leaf_lighting(image):
    """Reduce flash/shadow effects while preserving leaf color details."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    rgb = np.array(image)

    # Gray-world white balance keeps flash color casts from dominating.
    rgb_float = rgb.astype(np.float32)
    channel_means = rgb_float.reshape(-1, 3).mean(axis=0)
    gray_mean = channel_means.mean()
    scale = gray_mean / np.maximum(channel_means, 1.0)
    balanced = np.clip(rgb_float * scale, 0, 255).astype(np.uint8)

    # CLAHE on luminance lifts shadows and softens harsh local contrast.
    lab = cv2.cvtColor(balanced, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    corrected = cv2.merge((l_channel, a_channel, b_channel))
    corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)

    return Image.fromarray(corrected)


def gamma_correct(image, gamma):
    """Adjust image brightness for prediction-time lighting robustness."""
    rgb = np.array(image.convert('RGB'))
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return Image.fromarray(cv2.LUT(rgb, table))


def image_to_model_array(image):
    resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
    image = image.resize((224, 224), resample_filter)
    img_array = np.array(image).astype('float32')
    img_array = preprocess_input(img_array)
    return img_array


def preprocess_image(image):
    """Preprocess image for model prediction without changing the saved model."""
    try:
        image = ImageOps.exif_transpose(image).convert('RGB')
        lighting_corrected = correct_leaf_lighting(image)
        candidates = [
            image,
            lighting_corrected,
            gamma_correct(lighting_corrected, 0.85),
            gamma_correct(lighting_corrected, 1.15),
        ]

        return np.stack([image_to_model_array(candidate) for candidate in candidates], axis=0)
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def predict_disease(image):
    """Make prediction on the image"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction. Average lighting variants to reduce flash/shadow noise.
        predictions = model.predict(processed_image, verbose=0)
        if predictions.ndim == 2 and predictions.shape[0] > 1:
            predictions = np.mean(predictions, axis=0, keepdims=True)

        # Get the predicted class index and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        # Get the predicted class name
        predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f"Class_{predicted_class_idx}"

        # Determine if healthy or diseased
        is_healthy = predicted_class.lower() == 'healthy'
        disease_status = 'healthy' if is_healthy else 'diseased'

        # Check confidence threshold for unknown cases
        if confidence < 0.3:
            disease_status = 'unknown'
            disease_name = 'Unable to determine - image may not be from trained dataset or poor quality'
        elif len(class_names) == 2 and not is_healthy:
            disease_name = 'Disease detected'
        elif not is_healthy:
            disease_name = predicted_class.split('___', 1)[1] if '___' in predicted_class else predicted_class
        else:
            disease_name = 'None (Plant is healthy)'

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                "class": class_names[idx] if idx < len(class_names) else f"Class_{idx}",
                "confidence": float(predictions[0][idx]),
                "status": "healthy" if "healthy" in (class_names[idx] if idx < len(class_names) else f"Class_{idx}").lower() else "diseased"
            }
            for idx in top_3_indices
        ]

        suggestion = get_solution_suggestion(predicted_class, disease_status)

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "disease_status": disease_status,
            "disease_name": disease_name,
            "suggestion": suggestion,
            "top_3_predictions": top_3_predictions
        }

    except Exception as e:
        raise ValueError(f"Error making prediction: {str(e)}")


def get_solution_suggestion(predicted_class, disease_status):
    if disease_status == 'healthy':
        return DISEASE_SUGGESTIONS['healthy']
    if disease_status == 'unknown':
        return DISEASE_SUGGESTIONS['unknown']

    if predicted_class in DISEASE_SUGGESTIONS:
        return DISEASE_SUGGESTIONS[predicted_class]

    return DISEASE_SUGGESTIONS['diseased']


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_count": len(class_names)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict plant disease from uploaded image"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']

        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed. Use: png, jpg, jpeg, bmp, tiff"}), 400

        # Read and process the image
        try:
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

        # Make prediction
        result = predict_disease(image)

        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "disease_status": result["disease_status"],
            "disease_name": result["disease_name"],
            "suggestion": result["suggestion"],
            "top_3_predictions": result["top_3_predictions"]
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of all disease classes"""
    return jsonify({
        "classes": class_names,
        "count": len(class_names)
    })

@app.route('/', methods=['GET'])
def index():
    """Serve the web interface"""
    return send_from_directory('static', 'index.html')

@app.route('/api')
def api_info():
    """API information and documentation"""
    return jsonify({
        "name": "Plant Disease Detection API",
        "version": "1.0.0",
        "description": "API for detecting plant diseases using deep learning",
        "endpoints": {
            "GET /health": "Check API health and model status",
            "POST /predict": "Upload image and get disease prediction",
            "GET /classes": "Get list of all disease classes",
            "GET /api": "API information",
            "GET /": "Web interface"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS)
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("🚀 Starting Plant Disease Detection API...")
        print("🌐 Web Interface: http://localhost:5000")
        print("📡 API Endpoints: http://localhost:5000/api")
        print("📖 API Documentation: http://localhost:5000/api")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please check model path and try again.")
