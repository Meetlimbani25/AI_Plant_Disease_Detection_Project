# Plant Disease Detection API

A Flask REST API for detecting plant diseases from leaf images using a trained TensorFlow/Keras model.

## Overview

This project provides:

- A Flask server in `app.py` for image uploads and prediction
- A preprocessed model loader for `model/plant_model.h5`
- Image preprocessing and prediction endpoints
- Support for 38 plant disease classes from the PlantVillage dataset

## Features

- Upload plant leaf images for disease prediction
- Returns the predicted disease class, confidence, and top 3 predictions
- Supports PNG, JPG, JPEG, BMP, and TIFF image formats
- Includes `/health` and `/classes` endpoints
- Uses CORS for browser-based clients

## Requirements

- Python 3.8 or higher
- TensorFlow
- Flask
- Pillow
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `app.py` — Flask application and inference API
- `train_model.py` — training and model-building logic
- `requirements.txt` — Python dependencies
- `Dataset/` — local dataset files (ignored by Git)
- `model/` — saved trained model files (ignored by Git)
- `uploads/` — temporary upload folder
- `static/` — optional static web files

## Setup

### 1. Download or place the trained model

The app expects the model at:

```text
model/plant_model.h5
```

If you do not have a pre-trained model, you can train one using `train_model.py`. The `model/` folder is included in `.gitignore`, so large model files are not stored in the repository.

### 2. (Optional) Download the dataset for training

The repository can consume the PlantVillage dataset at:

```text
Dataset/PlantVillage
```

This dataset is not required to run the API if you already have `model/plant_model.h5`.

If you want to retrain the model, download PlantVillage from Kaggle. Example:

```bash
pip install kaggle
kaggle datasets download -d mohitsingh1804/plantvillage -p Dataset --unzip
```

### 3. Run the API

```bash
python app.py
```

The service runs at `http://localhost:5000`.

## API Endpoints

### GET /health

Returns service health and model load status.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes_count": 38
}
```

### GET /classes

Returns all supported disease class names.

Example response:

```json
{
  "classes": [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    ...
  ],
  "count": 38
}
```

### POST /predict

Upload an image file and receive disease prediction results.

Form field:

- `image`: image file (PNG, JPG, JPEG, BMP, TIFF)

Example response:

```json
{
  "success": true,
  "prediction": "Tomato___healthy",
  "confidence": 0.9876,
  "disease_status": "healthy",
  "disease_name": "None (Plant is healthy)",
  "suggestion": "Your plant appears healthy. Continue regular watering, balanced nutrition, and good air circulation.",
  "top_3_predictions": [
    {
      "class": "Tomato___healthy",
      "confidence": 0.9876,
      "status": "healthy"
    },
    {
      "class": "Tomato___Late_blight",
      "confidence": 0.0089,
      "status": "diseased"
    },
    {
      "class": "Tomato___Early_blight",
      "confidence": 0.0035,
      "status": "diseased"
    }
  ]
}
```

## Usage Examples

### Python

```python
import requests

response = requests.get('http://localhost:5000/health')
print(response.json())

with open('plant_leaf.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    print(response.json())
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('image', imageFile);

fetch('http://localhost:5000/predict', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

### cURL

```bash
curl -X POST -F "image=@plant_leaf.jpg" http://localhost:5000/predict
```

## Training the Model

If you need to train or retrain the model:

```bash
python train_model.py
```

This script collects images from local dataset folders and builds a binary classifier for healthy vs diseased leaves.

## Notes

- The dataset directory and model files are intentionally excluded from Git via `.gitignore`.
- For live inference, you only need the saved model file (`model/plant_model.h5`) and the Flask app.
- The dataset is only required when training or updating the model.

## License

MIT License
