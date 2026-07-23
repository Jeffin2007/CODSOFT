"""
Flask Web Server & REST API for VisionID Face Detection & Recognition Platform
"""

import os
import time
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from face_engine import EngineConfig, FacePipeline
from face_engine.utils import MediaHandler

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Initialize Engine Pipeline
config = EngineConfig(detector_type="yunet")
pipeline = FacePipeline(config)

UPLOAD_FOLDER = os.path.join(config.data_dir, "uploads")
OUTPUT_FOLDER = os.path.join(config.data_dir, "output")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/detect", methods=["POST"])
def detect_image():
    data = request.get_json(force=True)
    b64_img = data.get("image", "")
    
    if not b64_img:
        return jsonify({"status": "error", "message": "No image payload provided"}), 400

    cv_img = MediaHandler.b64_to_cv2(b64_img)
    if cv_img is None:
        return jsonify({"status": "error", "message": "Invalid Base64 image payload"}), 400

    ann_img, detections, telemetry = pipeline.process(cv_img, draw_annotations=True)
    out_b64 = MediaHandler.cv2_to_b64(ann_img, format=".jpg")

    return jsonify({
        "status": "success",
        "annotated_image": out_b64,
        "detections": detections,
        "telemetry": telemetry
    })

@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    """
    High-speed endpoint for live webcam canvas frames
    """
    data = request.get_json(force=True)
    b64_img = data.get("image", "")

    if not b64_img:
        return jsonify({"status": "error", "message": "Empty frame"}), 400

    cv_img = MediaHandler.b64_to_cv2(b64_img)
    if cv_img is None:
        return jsonify({"status": "error", "message": "Decode error"}), 400

    ann_img, detections, telemetry = pipeline.process(cv_img, draw_annotations=True)
    out_b64 = MediaHandler.cv2_to_b64(ann_img, format=".jpg")

    return jsonify({
        "status": "success",
        "processed_image": out_b64,
        "detections": detections,
        "telemetry": telemetry
    })

@app.route("/api/enroll", methods=["POST"])
def enroll_identity():
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    b64_img = data.get("image", "")

    if not name or not b64_img:
        return jsonify({"status": "error", "message": "Name and image are required"}), 400

    cv_img = MediaHandler.b64_to_cv2(b64_img)
    if cv_img is None:
        return jsonify({"status": "error", "message": "Could not decode face photo"}), 400

    success = pipeline.enroll_face(cv_img, name=name)
    if success:
        return jsonify({"status": "success", "message": f"Enrolled identity: '{name}'"})
    else:
        return jsonify({"status": "error", "message": "No face detected in photo for enrollment"}), 400

@app.route("/api/identities", methods=["GET"])
def get_identities():
    identities_list = []
    gallery_data = pipeline.gallery.gallery
    for name, vec_list in gallery_data.items():
        identities_list.append({
            "name": name,
            "sample_count": len(vec_list)
        })
    return jsonify({"status": "success", "identities": identities_list})

@app.route("/api/identities/<name>", methods=["DELETE"])
def delete_identity(name):
    success = pipeline.gallery.remove_identity(name)
    if success:
        return jsonify({"status": "success", "message": f"Deleted identity '{name}'"})
    return jsonify({"status": "error", "message": f"Identity '{name}' not found"}), 440

@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.get_json(force=True)
    detector = data.get("detector_type")
    if detector:
        pipeline.switch_detector(detector)

    if "det_confidence_threshold" in data:
        pipeline.config.det_confidence_threshold = float(data["det_confidence_threshold"])
        pipeline.detector.confidence_threshold = pipeline.config.det_confidence_threshold

    if "rec_similarity_threshold" in data:
        pipeline.config.rec_similarity_threshold = float(data["rec_similarity_threshold"])

    return jsonify({"status": "success", "config": {
        "detector_type": pipeline.config.detector_type,
        "det_confidence_threshold": pipeline.config.det_confidence_threshold,
        "rec_similarity_threshold": pipeline.config.rec_similarity_threshold
    }})

@app.route("/api/process_video", methods=["POST"])
def process_video_endpoint():
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file attached"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected video file"}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_filename = f"processed_{int(time.time())}_{filename}"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    file.save(input_path)

    def frame_cb(frame):
        ann_frame, det_list, telem = pipeline.process(frame)
        return ann_frame, {"face_count": len(det_list)}

    res = MediaHandler.process_video_file(input_path, output_path, frame_cb)

    return jsonify({
        "status": "success",
        "output_url": f"/media/output/{output_filename}",
        "metrics": res
    })

@app.route("/media/output/<filename>")
def serve_output_media(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    print("[INFO] Starting VisionID Face Engine Web App on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
