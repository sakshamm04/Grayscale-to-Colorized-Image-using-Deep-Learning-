# colorizer.py
import os
import cv2
import numpy as np
from cv2 import dnn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests

# Use the writable /tmp directory for the downloaded model
DATA_DIR = "/tmp/models"

def load_model():
    global net, reg_model
    
    model_name = "colorization_release_v2.caffemodel"
    model_path = os.path.join(DATA_DIR, model_name)
    proto_path = os.path.join('models', 'colorization_deploy_v2.prototxt')
    hull_path = os.path.join('models', 'pts_in_hull.npy')
    download_url = "https://drive.google.com/uc?export=download&id=1vbxBPDvkKtIZUKM0bAi7NbWeZbohRnw1"

    if not os.path.exists(model_path):
        print(f"'{model_name}' not found. Downloading to '{DATA_DIR}'...")
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded successfully!")
        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading model: {e}")
            exit()
    
    print("⏳ Initializing colorizer engine...")
    try:
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

# ... (The rest of your functions: process_image, etc., are correct and don't need changes)
def process_image(image_path, net):
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
    features = cv2.calcHist([colorized_image_obj], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    predicted_metrics = reg_model.predict([features])
    ssim_pred, psnr_pred = predicted_metrics[0]
    return ssim_pred, psnr_pred

def save_rgb_histogram(image_obj, save_path):
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