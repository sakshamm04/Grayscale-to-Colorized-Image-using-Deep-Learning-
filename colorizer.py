# colorizer.py
import os
import cv2
import numpy as np
from cv2 import dnn
import joblib
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt

# --- LOAD MODELS (These are loaded when the module is imported) ---
print("⏳ Initializing colorizer engine...")

# Caffe Model for Colorization
proto_path = os.path.join('models', 'colorization_deploy_v2.prototxt')
model_path = os.path.join('models', 'colorization_release_v2.caffemodel')
hull_path = os.path.join('models', 'pts_in_hull.npy')

net = dnn.readNetFromCaffe(proto_path, model_path)
kernel = np.load(hull_path)

# Add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Regression Model for Accuracy Prediction
try:
    reg_model = joblib.load('regression_model.pkl')
except FileNotFoundError:
    reg_model = None # Handle case where model isn't trained yet
    print("⚠️ Warning: 'regression_model.pkl' not found. Run train.py to create it.")

print("✅ Colorizer engine ready!")


# --- CORE FUNCTIONS ---

def process_image(image_path):
    """Takes a path to an image and returns the colorized CV2 image object."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Convert to LAB color space
    scaled = image.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Extract the L channel, resize, and process for the model
    L = cv2.split(lab_img)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(cv2.resize(L, (224, 224))))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the predicted 'ab' channel to the original image size
    ab_channel = cv2.resize(ab_channel, (image.shape[1], image.shape[0]))

    # Combine the original L channel with the predicted ab channel
    L = cv2.split(lab_img)[0]
    colorized_lab = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

    # Convert back to BGR color space
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_bgr = np.clip(colorized_bgr, 0, 1)
    colorized_bgr = (255 * colorized_bgr).astype("uint8")
    return colorized_bgr

def predict_quality_metrics(colorized_image_obj):
    """Takes a colorized image object and predicts its quality metrics."""
    if reg_model is None:
        return 0.0, 0.0 # Return default values if model not trained

    features = cv2.calcHist([colorized_image_obj], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    predicted_metrics = reg_model.predict([features])
    ssim_pred, psnr_pred = predicted_metrics[0]
    return ssim_pred, psnr_pred

def save_rgb_histogram(image_obj, save_path):
    """Generates and saves an RGB histogram plot for the given image."""
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
    plt.close() # Important: close the plot to free up memory