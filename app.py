import os
import uuid
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import cv2
from waitress import serve
import base64
import io
import numpy as np
from colorizer import load_model, process_image_array, predict_quality_metrics, save_rgb_histogram

# Initialize Flask app
app = Flask(__name__)

# Global variables for models
net = None
reg_model = None

def initialize_models():
    """Load models at startup"""
    global net, reg_model
    print("🚀 Starting AI Image Colorizer...")
    net, reg_model = load_model()
    
    if net is None or reg_model is None:
        print("❌ Failed to load models. App cannot start.")
        return False
    
    print("✅ Models loaded successfully!")
    return True

@app.route('/', methods=['GET', 'POST'])
def index():
    global net, reg_model
    
    if request.method == 'POST':
        # Check if models are loaded
        if net is None or reg_model is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        file = request.files.get('file')
        if not file or not file.filename:
            return jsonify({'error': 'No file selected'}), 400

        try:
            # Read image data
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Could not decode image'}), 400
            
            # Process image
            colorized_img = process_image_array(image, net)
            ssim, psnr = predict_quality_metrics(colorized_img, reg_model)
            
            # Convert images to base64 for display
            _, original_buffer = cv2.imencode('.jpg', image)
            original_b64 = base64.b64encode(original_buffer).decode('utf-8')
            
            _, colorized_buffer = cv2.imencode('.jpg', colorized_img)
            colorized_b64 = base64.b64encode(colorized_buffer).decode('utf-8')
            
            # Generate histogram
            histogram_path = f'/tmp/hist_{uuid.uuid4().hex}.png'
            save_rgb_histogram(colorized_img, histogram_path)
            
            with open(histogram_path, 'rb') as f:
                histogram_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Clean up temp file
            os.remove(histogram_path)
            
            return render_template('index.html',
                                   success=True,
                                   original_image_b64=original_b64,
                                   colorized_image_b64=colorized_b64,
                                   histogram_b64=histogram_b64,
                                   ssim=f"{ssim:.4f}",
                                   psnr=f"{psnr:.2f}")
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    return render_template('index.html', success=False)

@app.route('/health')
def health():
    """Health check endpoint"""
    global net, reg_model
    if net is not None and reg_model is not None:
        return jsonify({'status': 'healthy', 'models_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'models_loaded': False}), 500

if __name__ == '__main__':
    # Initialize models first
    if not initialize_models():
        print("❌ Cannot start app without models")
        exit(1)
    
    # Start the Flask server
    port = int(os.environ.get('PORT', 7860))
    print(f"🌐 Starting server on 0.0.0.0:{port}")
    
    # Use waitress for production
    serve(app, host='0.0.0.0', port=port)
