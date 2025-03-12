import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import concurrent.futures

app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_symmetry(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to load image.", None

        height, width = image.shape[:2]
        if width > 240:
            scale = 240 / width
            image = cv2.resize(image, (240, int(height * scale)))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, "No face detected.", None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        landmarks = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])

        nose_tip = landmarks[1]
        mid_x = nose_tip[0]

        left_landmarks = landmarks[:234]
        right_landmarks = landmarks[234:468]
        right_landmarks_mirrored = np.array([[mid_x - (x - mid_x), y] for x, y in right_landmarks])

        differences = np.linalg.norm(left_landmarks - right_landmarks_mirrored, axis=1)
        symmetry_percentage = 100 - (np.mean(differences) / mid_x * 100)

        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        annotated_image_path = os.path.join(UPLOAD_FOLDER, "annotated_" + os.path.basename(image_path))
        cv2.imwrite(annotated_image_path, image)

        return symmetry_percentage, annotated_image_path, None
    except Exception as e:
        return None, str(e), None

@app.route("/", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."})

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format. Please upload a PNG, JPG, or JPEG image."})

    filename = str(uuid.uuid4()) + "_" + file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(calculate_symmetry, file_path)
            symmetry_percentage, annotated_image_path, error = future.result(timeout=30)
    except concurrent.futures.TimeoutError:
        return jsonify({"error": "Image processing timed out."})

    if error:
        return jsonify({"error": error})

    return jsonify({
        "symmetry_percentage": symmetry_percentage,
        "annotated_image": os.path.basename(annotated_image_path)
    })

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
