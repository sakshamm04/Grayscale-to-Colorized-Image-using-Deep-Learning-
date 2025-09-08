# colorizer.py
import os
import cv2
import numpy as np
from cv2 import dnn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests # <--- Make sure 'requests' is in your requirements.txt

# --- MODEL DOWNLOAD AND LOAD LOGIC ---

# Define paths and the direct download URL
model_dir = "models"
model_name = "colorization_release_v2.caffemodel"
model_path = os.path.join(model_dir, model_name)
proto_path = os.path.join(model_dir, 'colorization_deploy_v2.prototxt')
hull_path = os.path.join(model_dir, 'pts_in_hull.npy')

# Your direct Google Drive download link
download_url = "https://drive.google.com/uc?export=download&id=1vbxBPDvkKtIZUKM0bAi7NbWeZbohRnw1"

# Check if the model file exists, and download it if it doesn't
if not os.path.exists(model_path):
    print(f"'{model_name}' not found. Downloading from Google Drive...")
    os.makedirs(model_dir, exist_ok=True)
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Model downloaded successfully!")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading model: {e}")
        exit() # Exit if the download fails

# --- LOAD MODELS (The rest of the file is the same) ---
print("⏳ Initializing colorizer engine...")

try:
    net = dnn.readNetFromCaffe(proto_path, model_path)
    kernel = np.load(hull_path)

    # Add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    reg_model = joblib.load('regression_model.pkl')
    print("✅ Colorizer engine ready!")

except cv2.error as e:
    print(f"❌ OpenCV Error: Failed to load model files. Make sure .prototxt and .npy files are present. Error: {e}")
    exit()
except FileNotFoundError as e:
    print(f"❌ FileNotFoundError: Could not find a necessary file. Error: {e}")
    exit()


# --- CORE FUNCTIONS ---
def process_image(image_path):
    # ... (rest of your functions remain exactly the same) ...
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
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

def predict_quality_metrics(colorized_image_obj):
    if reg_model is None:
        return 0.0, 0.0
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