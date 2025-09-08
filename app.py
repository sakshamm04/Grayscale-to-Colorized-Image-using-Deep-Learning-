import os
import cv2
import numpy as np
from cv2 import dnn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model paths - using Git LFS files
MODEL_CONFIG = {
    'caffe_model': {
        'local_path': 'models/colorization_release_v2.caffemodel',
        'filename': 'colorization_release_v2.caffemodel'
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

def validate_model_file(filepath, min_size=1024):
    """Validate that model file exists and is not corrupted"""
    if not os.path.exists(filepath):
        print(f"❌ Model file not found: {filepath}")
        return False

    file_size = os.path.getsize(filepath)

    # Check if file is empty or too small
    if file_size < min_size:
        print(f"❌ {filepath} is too small ({file_size} bytes). May be corrupted or not downloaded from Git LFS.")
        return False

    # Try to read file to ensure it's accessible
    try:
        with open(filepath, 'rb') as f:
            f.read(min_size)  # Read first chunk
        return True
    except Exception as e:
        print(f"❌ {filepath} appears to be corrupted: {e}")
        return False

def check_git_lfs_setup():
    """Check if Git LFS files are properly downloaded"""
    model_path = MODEL_CONFIG['caffe_model']['local_path']

    if not os.path.exists(model_path):
        print(f"❌ Caffe model not found at {model_path}")
        print("💡 Make sure you've cloned the repository with Git LFS enabled:")
        print("   git lfs pull")
        print("   or git clone <repo> && cd <repo> && git lfs pull")
        return False

    # Check if it's a Git LFS pointer file (small text file)
    file_size = os.path.getsize(model_path)
    if file_size < 1024 * 1024:  # Less than 1MB
        print(f"❌ Model file appears to be a Git LFS pointer ({file_size} bytes)")
        print("💡 Run 'git lfs pull' to download the actual model files")
        return False

    print(f"✅ Git LFS model file validated ({file_size / (1024*1024):.1f} MB)")
    return True

def load_caffe_model():
    """Load the Caffe model from Git LFS files"""
    # Paths to model files
    model_path = MODEL_CONFIG['caffe_model']['local_path']
    prototxt_path = MODEL_CONFIG['prototxt']['local_path']
    hull_path = MODEL_CONFIG['hull_points']['local_path']

    print("🔄 Loading models from Git LFS files...")

    # Check if Git LFS files are properly downloaded
    if not check_git_lfs_setup():
        return None

    # Validate all required files
    required_files = [
        (prototxt_path, 1024, "prototxt file"),
        (model_path, 100 * 1024 * 1024, "caffe model"),  # Should be ~129MB
        (hull_path, 1024, "hull points file")
    ]

    for filepath, min_size, description in required_files:
        if not validate_model_file(filepath, min_size):
            print(f"❌ {description} validation failed: {filepath}")
            return None

    try:
        print("🔄 Initializing Caffe neural network...")
        net = dnn.readNetFromCaffe(prototxt_path, model_path)

        # Load hull points
        print("🔄 Loading color space points...")
        kernel = np.load(hull_path)

        # Configure network layers
        print("🔄 Configuring network layers...")
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = kernel.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

        print("✅ Caffe model loaded successfully!")
        return net

    except Exception as e:
        print(f"❌ Failed to load Caffe model: {e}")
        print("💡 Possible issues:")
        print("   - Git LFS files not downloaded (run 'git lfs pull')")
        print("   - Corrupted model files")
        print("   - OpenCV DNN module not properly installed")
        return None

def load_regression_model():
    """Load the regression model"""
    reg_path = MODEL_CONFIG['regression_model']['local_path']

    if not os.path.exists(reg_path):
        print(f"❌ Regression model not found: {reg_path}")
        print("💡 Make sure regression_model.pkl is in the repository root")
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
    Main function to load all models from Git LFS files.
    Returns (net, reg_model) tuple or (None, None) if loading fails.
    """
    print("🚀 Starting model loading from Git LFS...")

    try:
        # Load Caffe model from Git LFS
        net = load_caffe_model()
        if net is None:
            print("❌ Failed to load Caffe model from Git LFS")
            return None, None

        # Load regression model  
        reg_model = load_regression_model()
        if reg_model is None:
            print("❌ Failed to load regression model")
            return None, None

        print("🎉 All models loaded successfully from Git LFS!")
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

def verify_git_lfs_files():
    """Verify all Git LFS files are properly downloaded"""
    print("🔍 Verifying Git LFS files...")

    model_path = MODEL_CONFIG['caffe_model']['local_path']
    expected_size = 100 * 1024 * 1024  # ~100MB minimum

    if not os.path.exists(model_path):
        print(f"❌ Model file missing: {model_path}")
        return False

    actual_size = os.path.getsize(model_path)
    if actual_size < expected_size:
        print(f"❌ Model file too small: {actual_size} bytes (expected > {expected_size})")
        print("💡 Run 'git lfs pull' to download actual files")
        return False

    print(f"✅ Git LFS files verified ({actual_size / (1024*1024):.1f} MB)")
    return True

if __name__ == "__main__":
    # Test Git LFS and model loading
    print("🧪 Testing Git LFS setup and model loading...")

    if not verify_git_lfs_files():
        print("❌ Git LFS verification failed!")
        exit(1)

    net, reg_model = load_model()

    if net is not None and reg_model is not None:
        print("✅ All tests passed! Models ready for use.")
    else:
        print("❌ Model loading test failed!")
        exit(1)
