# app.py
import os
from flask import Flask, request, render_template
import cv2
from waitress import serve # <--- ADD THIS IMPORT

# Import our custom functions from the colorizer.py file
from colorizer import process_image, predict_quality_metrics, save_rgb_histogram, load_model

# --- FLASK APP SETUP ---
app = Flask(__name__)

# Define folder paths
UPLOAD_FOLDER = os.path.join('static', 'uploads')
COLORIZED_FOLDER = os.path.join('static', 'colorized')
PLOT_FOLDER = os.path.join('static', 'plots')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COLORIZED_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# --- LOAD MODEL AT STARTUP ---
# With Waitress, we can safely load the model once when the app starts.
load_model()

# --- FLASK ROUTE (The main webpage) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename:
            return "No file selected!", 400

        # 1. Save the uploaded image
        filename = file.filename
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        # 2. Process the image using our imported function
        colorized_img_obj = process_image(upload_path)
        colorized_filename = 'colorized_' + filename
        colorized_path = os.path.join(COLORIZED_FOLDER, colorized_filename)
        cv2.imwrite(colorized_path, colorized_img_obj)

        # 3. Predict metrics using our imported function
        ssim, psnr = predict_quality_metrics(colorized_img_obj)

        # 4. Generate histogram using our imported function
        plot_filename = 'plot_' + filename
        plot_path = os.path.join(PLOT_FOLDER, plot_filename)
        save_rgb_histogram(colorized_img_obj, plot_path)

        # 5. Render the results page
        return render_template('index.html',
                               original_image=filename,
                               colorized_image=colorized_filename,
                               plot_image=plot_filename,
                               ssim=f"{ssim:.4f}",
                               psnr=f"{psnr:.2f}")

    return render_template('index.html')

# --- RUN THE APP ---
if __name__ == '__main__':
    # Use Waitress to serve the app
    serve(app, host='0.0.0.0', port=8080)