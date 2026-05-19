from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    try:
        image_data = request.files['image']
        image_array = np.frombuffer(image_data.read(), np.uint8) # Recebe imagem do frontend ( camera.html ) 
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False) #Modelo deepface vai analizar a imagem e trazer a emocao mais forte
        emotion = result[0]['dominant_emotion']                         
        emotions = result[0]['emotion']
        
        return jsonify({'emotion': emotion, 'all_emotions': emotions})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
