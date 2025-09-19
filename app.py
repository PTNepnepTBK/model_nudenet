
from flask import Flask, request, jsonify
from nudenet import NudeDetector
from werkzeug.utils import secure_filename
from io import BytesIO
import cv2
import numpy as np
import os
import requests

app = Flask(__name__)

# Fungsi untuk download file dari Google Drive
def download_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Cek dan download model jika tidak ada
MODEL_PATH = "640m.onnx"
if not os.path.exists(MODEL_PATH):
    print("Model file not found. Downloading from Google Drive...")
    try:
        # Google Drive file ID dari URL
        file_id = "1DF9b21MrgbWV2Zg0sqQBw6vsAgmgugyq"
        download_from_google_drive(file_id, MODEL_PATH)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Failed to download model: {e}")
        exit(1)
else:
    print("model exist")

detector = NudeDetector(model_path=MODEL_PATH, inference_resolution=640)

# Konfigurasi upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect', methods=['POST'])
def detect_image():
    # Cek apakah ada file yang diupload
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Cek apakah file valid
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use: png, jpg, jpeg, gif, bmp'}), 400
    
    try:
        # Baca gambar ke memory buffer
        image_stream = BytesIO(file.read())
        
        # Convert ke numpy array untuk OpenCV
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format or corrupted file'}), 400
        
        # Deteksi NSFW langsung dari array numpy
        results = detector.detect(image)
        
        return jsonify({
            'filename': secure_filename(file.filename),
            'results': results,
            'status': 'success'
        })
            
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/')
def home():
    return '''
    <h2>NSFW Detector API</h2>
    <p>Upload an image to detect NSFW content</p>
    <form action="/detect" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Detect NSFW</button>
    </form>
    <p>Or use POST /detect with 'image' file in form-data</p>
    '''

if __name__ == '__main__':
	app.run(debug=True)
