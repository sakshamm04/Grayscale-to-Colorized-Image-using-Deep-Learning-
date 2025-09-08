import os
import cv2
import numpy as np
from cv2 import dnn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
from urllib.parse import urlparse
import hashlib

# Configuration for model storage
MODELS_DIR = "/tmp/models"
CACHE_DIR = "/tmp/cache"

# Model URLs and file info
MODEL_CONFIG = {
    'caffe_model': {
        'url': 'https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1',
        'filename': 'colorization_release_v2.caffemodel',
        'expected_size': 128941848  # bytes, for validation
    },
    'prototxt': {
        'local_path': 'models/colorization_deploy_v2.prototxt',
        'filename': 'colorization_deploy_v2.prototxt'
    },
    'hull_points': {
        'local_path': 'models/pts_in_hull.npy', 
        'filename': 'pts_in_hull.npy'
    },
    'regression_model': {
        'local_path': 'regression_model.pkl',
        'filename': 'regression_model.pkl'
    }
}

def ensure_directories():
    """Create necessary directories"""
    for directory in [MODELS_DIR, CACHE_DIR]:
        os.makedirs(directory, exist_ok=True)

def get_file_hash(filepath):
    """Get SHA256 hash of file for validation"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return None

def download_file_with_progress(url, filepath):
    """Download file with better error handling and progress"""
    try:
        print(f"📥 Downloading {os.path.basename(filepath)}...")

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r📥 Progress: {percent:.1f}%", end="", flush=True)

        print(f"\n✅ Downloaded {os.path.basename(filepath)} successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def validate_model_file(filepath, expected_size=None):
    """Validate that model file is complete and not corrupted"""
    if not os.path.exists(filepath):
        return False

    file_size = os.path.getsize(filepath)

    # Check if file is empty
    if file_size == 0:
        print(f"❌ {filepath} is empty")
        return False

    # Check expected size if provided
    if expected_size and abs(file_size - expected_size) > 1024:  # Allow 1KB tolerance
        print(f"❌ {filepath} size mismatch. Expected: {expected_size}, Got: {file_size}")
        return False

    # Try to read file to ensure it's not corrupted
    try:
        with open(filepath, 'rb') as f:
            f.read(1024)  # Read first 1KB
        return True
    except:
        print(f"❌ {filepath} appears to be corrupted")
        return False

def load_caffe_model():
    """Download and load the Caffe model with better error handling"""
    ensure_directories()

    # Paths
    model_path = os.path.join(MODELS_DIR, MODEL_CONFIG['caffe_model']['filename'])
    prototxt_path = MODEL_CONFIG['prototxt']['local_path']
    hull_path = MODEL_CONFIG['hull_points']['local_path']

    # Check if prototxt and hull files exist locally
    if not os.path.exists(prototxt_path):
        print(f"❌ Prototxt file not found: {prototxt_path}")
        return None

    if not os.path.exists(hull_path):
        print(f"❌ Hull points file not found: {hull_path}")
        return None

    # Download or validate Caffe model
    if not os.path.exists(model_path) or not validate_model_file(
        model_path, MODEL_CONFIG['caffe_model']['expected_size']
    ):
        print("🔄 Caffe model not found or invalid. Downloading...")

        if not download_file_with_progress(
            MODEL_CONFIG['caffe_model']['url'], model_path
        ):
            return None

        # Validate downloaded file
        if not validate_model_file(model_path, MODEL_CONFIG['caffe_model']['expected_size']):
            print("❌ Downloaded model file validation failed")
            return None

    try:
        print("🔄 Loading Caffe model...")
        net = dnn.readNetFromCaffe(prototxt_path, model_path)

        # Load hull points
        kernel = np.load(hull_path)

        # Configure network layers
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = kernel.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

        print("✅ Caffe model loaded successfully!")
        return net

    except Exception as e:
        print(f"❌ Failed to load Caffe model: {e}")
        return None

def load_regression_model():
    """Load the regression model with error handling"""
    reg_path = MODEL_CONFIG['regression_model']['local_path']

    if not os.path.exists(reg_path):
        print(f"❌ Regression model not found: {reg_path}")
        return None

    try:
        print("🔄 Loading regression model...")
        reg_model = joblib.load(reg_path)
        print("✅ Regression model loaded successfully!")
        return reg_model

    except Exception as e:
        print(f"❌ Failed to load regression model: {e}")
        return None

def load_model():
    """
    Main function to load all models with comprehensive error handling.
    Returns (net, reg_model) tuple or (None, None) if loading fails.
    """
    print("🚀 Starting model loading process...")

    try:
        # Load Caffe model
        net = load_caffe_model()
        if net is None:
            print("❌ Failed to load Caffe model")
            return None, None

        # Load regression model  
        reg_model = load_regression_model()
        if reg_model is None:
            print("❌ Failed to load regression model")
            return None, None

        print("🎉 All models loaded successfully!")
        return net, reg_model

    except Exception as e:
        print(f"❌ Critical error during model loading: {e}")
        return None, None

def process_image(image_path, net):
    """Process image for colorization - kept for compatibility"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        return process_image_array(image, net)

    except Exception as e:
        print(f"❌ Image processing error: {e}")
        raise

def process_image_array(image, net):
    """Process opencv image array for colorization"""
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    scaled = image.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    L = cv2.split(lab_img)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(cv2.resize(L, (224, 224))))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (image.shape[1], image.shape[0]))

    L = cv2.split(lab_img)[0]
    colorized_lab = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_bgr = np.clip(colorized_bgr, 0, 1)
    colorized_bgr = (255 * colorized_bgr).astype("uint8")

    return colorized_bgr

def predict_quality_metrics(colorized_image_obj, reg_model):
    """Predict quality metrics using regression model"""
    try:
        features = cv2.calcHist(
            [colorized_image_obj], [0, 1, 2], None, 
            [8, 8, 8], [0, 256, 0, 256, 0, 256]
        ).flatten()

        predicted_metrics = reg_model.predict([features])
        ssim_pred, psnr_pred = predicted_metrics[0]

        return ssim_pred, psnr_pred

    except Exception as e:
        print(f"❌ Quality prediction error: {e}")
        return 0.5, 20.0  # Return default values

def save_rgb_histogram(image_obj, save_path):
    """Generate and save RGB histogram"""
    try:
        plt.figure(figsize=(8, 4))
        colors = ('b', 'g', 'r')

        for i, color in enumerate(colors):
            hist = cv2.calcHist([image_obj], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)

        plt.xlim([0, 256])
        plt.title('RGB Histogram of Colorized Image')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        print(f"❌ Histogram generation error: {e}")

# Health check function
def check_model_health():
    """Check if models can be loaded successfully"""
    net, reg_model = load_model()
    return net is not None and reg_model is not None

if __name__ == "__main__":
    # Test model loading
    print("🧪 Testing model loading...")
    net, reg_model = load_model()

    if net is not None and reg_model is not None:
        print("✅ Model loading test successful!")
    else:
        print("❌ Model loading test failed!")
