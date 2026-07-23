# VisionID AI - Production Face Detection & Recognition Platform

A modular, high-performance Computer Vision platform for detecting and recognizing faces in images, video files, and real-time webcam feeds. Built with modern PyTorch deep learning models (FaceNet InceptionResNetV1), state-of-the-art detector backends (YuNet DNN, MTCNN, Haar Cascade), Cosine Similarity face matching, and an interactive Glassmorphism Web App.

---

## 🌟 Key Features

1. **Multi-Model Detection Engine**:
   - **YuNet DNN**: SOTA lightweight OpenCV ONNX deep learning detector with 5 facial landmarks (eyes, nose, mouth).
   - **MTCNN**: Multi-task Cascaded Neural Networks (PyTorch).
   - **Haar Cascade**: Lightweight CPU fallback.

2. **Face Recognition & Biometric Gallery**:
   - **FaceNet (InceptionResNetV1)**: Extracts 512-dimensional $L_2$-normalized feature embeddings.
   - **Cosine Similarity Matching**: Identity assignment with threshold tuning and confidence scoring.
   - **Dynamic Identity DB**: Register new face identities via image file upload or live webcam snapshot; multi-shot embedding averaging stored in `data/identities.json`.

3. **Visual Annotations & Telemetry**:
   - Rounded bounding box drawing with color-coded status (Green for Enrolled, Orange for Unknown).
   - Facial landmark keypoint dots (eyes, nose tip, mouth corners).
   - Real-time telemetry HUD overlay displaying FPS, per-face latency, and detection counts.

4. **Multiple Entry Points**:
   - **Web Dashboard**: Interactive Flask web application with live camera canvas, dropzone photo inspector, video batch processor, and identity gallery manager.
   - **CLI Tool**: Process images, batch videos, enroll faces, or open desktop OpenCV windows directly from terminal.
   - **Python API**: Clean modular import interface for custom integrations.

---

## 🚀 Quick Start

### 1. Installation
Ensure Python 3.10+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Automated Verification Demo
```bash
python demo.py
```
This runs synthetic test image generation, enrolling a sample identity, executing the detection & recognition pipeline, and saving annotated output to `data/output/demo_result.jpg`.

---

## 💻 Web Application

Launch the Web Dashboard server:
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

### Web Dashboard Features:
- **Live Camera Stream**: Real-time webcam processing with live telemetry.
- **Image Inspector**: Drag-and-drop photo analyzer.
- **Video Processor**: Process `.mp4` / `.avi` videos with progress tracking.
- **Face Enrollment**: Register new identities with custom name tags.
- **Identity Database**: Inspect and manage enrolled biometric profiles.

---

## 🛠️ CLI Usage

### Detect & Recognize Faces in an Image
```bash
python cli.py detect --input photo.jpg --output output.jpg --model yunet
```

### Process Video File
```bash
python cli.py process-video --input input.mp4 --output output.mp4 --model yunet
```

### Enroll New Identity
```bash
python cli.py enroll --name "Dr. Sarah Connor" --image sarah.jpg
```

### Launch Live Webcam Window
```bash
python cli.py webcam --model yunet
```

### List Enrolled Identities
```bash
python cli.py identities
```

---

## 📁 Repository Architecture

```text
Task-4/
├── face_engine/
│   ├── config.py              # Central config defaults and color themes
│   ├── pipeline.py            # Unified orchestration pipeline
│   ├── detectors/
│   │   ├── base.py            # Abstract Base Detector class
│   │   ├── yunet.py           # OpenCV DNN YuNet implementation
│   │   ├── mtcnn.py           # PyTorch MTCNN implementation
│   │   └── haar_cascade.py    # Haar Cascade detector implementation
│   ├── recognizers/
│   │   ├── base.py            # Base Embedder interface
│   │   ├── facenet.py         # FaceNet InceptionResNetV1 512d model
│   │   └── gallery.py          # Biometric Identity DB & Cosine similarity matcher
│   └── utils/
│       ├── visualization.py   # Modern overlays, bounding boxes, landmarks & HUD
│       └── media.py           # Video file processing, base64 & image loading
├── templates/
│   └── index.html             # Glassmorphism Web App HTML
├── static/
│   ├── css/style.css          # Glassmorphism design system
│   └── js/app.js              # Frontend streaming & canvas interactions
├── app.py                     # Flask REST API & Web Server
├── cli.py                     # Command Line Interface
├── demo.py                    # Validation runner script
├── requirements.txt           # Dependency requirements
└── README.md                  # System Documentation
```
