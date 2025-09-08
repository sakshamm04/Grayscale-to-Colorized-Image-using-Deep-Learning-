# train.py
import numpy as np
from sklearn.linear_model import LinearRegression
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import joblib
import cv2
import os

# Import the image processing function from our new colorizer file
from colorizer import process_image

# --- IMPORTANT: Make sure your 'images' folder is set up correctly ---
# This list should contain the paths to your 10 gray/color training pairs.
training_data = [
    ('images/gray1.jpg', 'images/color1.jpg'),
    ('images/gray2.jpg', 'images/color2.jpg'),
    ('images/gray3.jpg', 'images/color3.jpg'),
    ('images/gray4.jpg', 'images/color4.jpg'),
    ('images/gray5.jpg', 'images/color5.jpg'),
    ('images/gray6.jpg', 'images/color6.jpg'),
    ('images/gray7.jpg', 'images/color7.jpg'),
    ('images/gray8.jpg', 'images/color8.jpg'),
    ('images/gray9.jpg', 'images/color9.jpg'),
    ('images/gray10.jpg', 'images/color10.jpg'),
]

def extract_features_and_metrics(grayscale_path, color_path):
    """Helper function to get features and true metrics for training."""
    colorized = process_image(grayscale_path)
    original_color = cv2.imread(color_path)

    # Calculate the true SSIM and PSNR values for training
    ssim_value, _ = compare_ssim(cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY),
                                 cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY), full=True)
    psnr_value = compare_psnr(original_color, colorized)

    # Extract features from the colorized image
    colorized_hist = cv2.calcHist([colorized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    features = colorized_hist
    metrics = [ssim_value, psnr_value]
    return features, metrics

# --- Main Training Logic ---
if __name__ == '__main__':
    print("🚀 Starting regression model training...")
    X_train = []
    y_train = []

    for grayscale_path, color_path in training_data:
        print(f"Processing {grayscale_path}...")
        if not os.path.exists(grayscale_path) or not os.path.exists(color_path):
            print(f"⚠️ Warning: Skipping pair, file not found: {grayscale_path} or {color_path}")
            continue
        features, metrics = extract_features_and_metrics(grayscale_path, color_path)
        X_train.append(features)
        y_train.append(metrics)

    if not X_train:
        print("❌ Error: No training data was processed. Please check your file paths in 'training_data' list.")
    else:
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Train a Linear Regression model
        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(reg_model, 'regression_model.pkl')
        print("\n✅ Model training complete! 'regression_model.pkl' has been saved.")