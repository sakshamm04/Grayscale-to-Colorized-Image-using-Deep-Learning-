import os
import cv2
import numpy as np
from cv2 import dnn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_model():
    """
    Loads all ML models from the local 'models' folder and returns them.
    """
    print("⏳ Initializing colorizer engine from local files...")
    try:
        proto_path = os.path.join('models', 'colorization_deploy_v2.prototxt')
        model_path = os.path.join('models', 'colorization_release_v2.caffemodel')
        hull_path = os.path.join('models', 'pts_in_hull.npy')
        
        net = dnn.readNetFromCaffe(proto_path, model_path)
        kernel = np.load(hull_path)

        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = kernel.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

        reg_model = joblib.load('regression_model.pkl')
        print("✅ Colorizer engine ready!")
        return net, reg_model
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load models. {e}")
        return None, None

def process_image(image_path, net):
    """Processes an uploaded image and returns the colorized version."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"OpenCV could not read the image at path: {image_path}. It might be a format issue.")
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
    """Predicts quality metrics for a colorized image."""
    features = cv2.calcHist([colorized_image_obj], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    predicted_metrics = reg_model.predict([features])
    ssim_pred, psnr_pred = predicted_metrics[0]
    return ssim_pred, psnr_pred

def save_rgb_histogram(image_obj, save_path):
    """Saves an RGB histogram plot of an image."""
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
