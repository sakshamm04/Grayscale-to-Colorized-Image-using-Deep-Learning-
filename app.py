import os
import uuid
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import cv2
from waitress import serve
from colorizer import load_model, process_image, predict_quality_metrics, save_rgb_histogram

app = Flask(__name__)

# Use the writable /tmp directory for all temporary user files
UPLOAD_FOLDER = '/tmp/uploads'
COLORIZED_FOLDER = '/tmp/colorized'
PLOT_FOLDER = '/tmp/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COLORIZED_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# Load models at startup and store them in variables
net, reg_model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if net is None or reg_model is None:
            return "Error: Models are not loaded. Please check the server logs.", 500
        
        file = request.files.get('file')
        if not file or not file.filename:
            return "No file selected!", 400

        # Generate a safe and unique filename for the upload
        original_filename = secure_filename(file.filename)
        unique_id = uuid.uuid4().hex
        _, extension = os.path.splitext(original_filename)
        safe_filename = f"{unique_id}{extension}"
        upload_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(upload_path)
        
        colorized_img_obj = process_image(upload_path, net)
        ssim, psnr = predict_quality_metrics(colorized_img_obj, reg_model)
        
        colorized_filename = 'colorized_' + safe_filename
        colorized_path = os.path.join(COLORIZED_FOLDER, colorized_filename)
        cv2.imwrite(colorized_path, colorized_img_obj)
        
        plot_filename = 'plot_' + safe_filename
        plot_path = os.path.join(PLOT_FOLDER, plot_filename)
        save_rgb_histogram(colorized_img_obj, plot_path)

        return render_template('index.html',
                               original_image=safe_filename,
                               colorized_image=colorized_filename,
                               plot_image=plot_filename,
                               ssim=f"{ssim:.4f}",
                               psnr=f"{psnr:.2f}")

    return render_template('index.html')

# Routes to serve files from the /tmp directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/colorized/<filename>')
def colorized_file(filename):
    return send_from_directory(COLORIZED_FOLDER, filename)

@app.route('/plots/<filename>')
def plot_file(filename):
    return send_from_directory(PLOT_FOLDER, filename)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
