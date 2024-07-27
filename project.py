from flask import Flask, request, jsonify, render_template
import face_recognition
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import io

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png',}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_photo_from_poi(poi_image_bytes):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    nparr = np.frombuffer(poi_image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        photo = img[y:y+h, x:x+w]
        
        is_success, buffer = cv2.imencode(".jpg", photo)
        if is_success:
            photo_bytes = io.BytesIO(buffer).getvalue()
            return photo_bytes

    return None

def compare_images(user_image_bytes, poi_image_bytes):
    user_image = face_recognition.load_image_file(BytesIO(user_image_bytes))
    poi_image = face_recognition.load_image_file(BytesIO(poi_image_bytes))
    
    user_encoding = face_recognition.face_encodings(user_image)
    poi_encoding = face_recognition.face_encodings(poi_image)
    
    if len(user_encoding) > 0 and len(poi_encoding) > 0:
        results = face_recognition.compare_faces([poi_encoding[0]], user_encoding[0])
        return bool(results[0])
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify_identity():
    if 'poi_image' not in request.files or 'user_image' not in request.files:
        return jsonify({'error': 'Please provide both inputs'}), 400

    poi_image = request.files['poi_image']
    user_image = request.files['user_image']

    if not (allowed_file(poi_image.filename) and allowed_file(user_image.filename)):
        return jsonify({'error': 'Unsupported format. Please upload the images in JPG, JPEG or PNG format.'}), 400

    poi_image_bytes = poi_image.read()
    user_image_bytes = user_image.read()

    extracted_poi_image_bytes = extract_photo_from_poi(poi_image_bytes)
    
    if extracted_poi_image_bytes is None:
        return jsonify({'error': 'No photo region detected in POI'}), 400

    match = compare_images(user_image_bytes, extracted_poi_image_bytes)

    return jsonify({'match': match}), 200

if __name__ == '__main__':
    app.run(debug=True)
