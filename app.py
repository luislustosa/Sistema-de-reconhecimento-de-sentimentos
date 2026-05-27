from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.post("/detect-face")
def detect_face():
    file = request.files.get("image")
    if not file:
        return jsonify(error="Campo 'image' não foi enviado."), 400

    try:
 
        data = np.frombuffer(file.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify(error="Arquivo enviado não é uma imagem válida."), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        faces_list = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]
        primary = None
        if faces_list:
            primary = max(faces_list, key=lambda f: f["w"] * f["h"])

        return jsonify(primary=primary, faces=faces_list)

    except Exception as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
