import os
import cv2
import numpy as np
from cv2 import dnn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model paths
MODEL_PATHS = {
    'caffe_model': 'models/colorization_release_v2.caffemodel',
    'prototxt': 'models/colorization_deploy_v2.prototxt', 
    'hull_points': 'models/pts_in_hull.npy',
    'regression_model': 'regression_model.pkl'
}

def load_model():
    """Load all models for the Flask app"""
    print("🚀 Loading models from Git LFS...")
    
    # Check all files exist
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"❌ {name} not found: {path}")
            return None, None
        
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ {name}: {size_mb:.1f} MB")
    
    try:
        # Load Caffe model
        print("🔄 Loading Caffe neural network...")
        net = dnn.readNetFromCaffe(
            MODEL_PATHS['prototxt'], 
            MODEL_PATHS['caffe_model']
        )
        
        # Configure network
        kernel = np.load(MODEL_PATHS['hull_points'])
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = kernel.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        
        print("✅ Caffe model loaded!")
        
        # Load regression model
        print("🔄 Loading regression model...")
        reg_model = joblib.load(MODEL_PATHS['regression_model'])
        print("✅ Regression model loaded!")
        
        print("🎉 All models ready!")
        return net, reg_model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def process_image_array(image, net):
    """Colorize image array"""
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    scaled = image.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    L = cv2.split(lab_img)[0] - 50
    
    net.setInput(cv2.dnn.blobFromImage(cv2.resize(L, (224, 224))))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (image.shape[1], image.shape[0]))
    
    L = cv2.split(lab_img)[0]
    colorized_lab = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_bgr = np.clip(colorized_bgr, 0, 1)
    
    return (255 * colorized_bgr).astype("uint8")

def predict_quality_metrics(colorized_image, reg_model):
    """Predict quality metrics"""
    try:
        features = cv2.calcHist([colorized_image], [0, 1, 2], None, 
                               [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        metrics = reg_model.predict([features])[0]
        return metrics[0], metrics[1]  # SSIM, PSNR
    except:
        return 0.5, 20.0

def save_rgb_histogram(image, save_path):
    """Generate RGB histogram"""
    plt.figure(figsize=(8, 4))
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.xlim([0, 256])
    plt.title('RGB Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

# NO TEST CODE HERE! This was the problem.
