#!/usr/bin/env python3
"""
Plant Disease Detection API Test Script
This script demonstrates how to use the Plant Disease Detection API
"""

import requests
import json
import os
from pathlib import Path

API_BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing API Health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy!")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Classes count: {data['classes_count']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

def test_classes():
    """Test the classes endpoint"""
    print("\n📋 Getting disease classes...")
    try:
        response = requests.get(f"{API_BASE_URL}/classes")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['count']} disease classes")
            print("📋 Sample classes:")
            for i, cls in enumerate(data['classes'][:5]):
                print(f"   {i+1}. {cls}")
            if data['count'] > 5:
                print(f"   ... and {data['count'] - 5} more")
            return True
        else:
            print(f"❌ Classes endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

def test_prediction(image_path):
    """Test prediction with an image file"""
    print(f"\n🔬 Testing prediction with image: {image_path}")
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False

    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{API_BASE_URL}/predict", files=files)

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Prediction successful!")
                print(f"   Disease: {data['prediction']}")
                print(".2%")
                print("🏆 Top 3 predictions:")
                for i, pred in enumerate(data['top_3_predictions'], 1):
                    print(".2%")
            else:
                print(f"❌ Prediction failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Prediction request failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return False

    return True

def find_sample_image():
    """Try to find a sample image in the dataset"""
    dataset_path = Path("Dataset/PlantVillage/train")

    if not dataset_path.exists():
        return None

    # Look for any image file in the first disease class directory
    for disease_dir in dataset_path.iterdir():
        if disease_dir.is_dir():
            for image_file in disease_dir.iterdir():
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    return str(image_file)
    return None

def main():
    """Main test function"""
    print("🌱 Plant Disease Detection API Test")
    print("=" * 40)

    # Test health
    if not test_health():
        print("\n❌ API is not running. Please start the API first with: python app.py")
        return

    # Test classes
    test_classes()

    # Test prediction with sample image
    sample_image = find_sample_image()
    if sample_image:
        print(f"\n📸 Found sample image: {sample_image}")
        test_prediction(sample_image)
    else:
        print("\n📸 No sample image found in Dataset/PlantVillage/train/")
        print("💡 To test prediction, place a plant leaf image in the project directory")
        print("   and call test_prediction('your_image.jpg')")

    print("\n" + "=" * 40)
    print("🎉 API testing completed!")

if __name__ == "__main__":
    main()