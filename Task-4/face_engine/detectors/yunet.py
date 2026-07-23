"""
OpenCV DNN YuNet Face Detector Implementation
"""

import os
import cv2
import urllib.request
import numpy as np
from typing import List
from .base import BaseFaceDetector, DetectionResult

YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

class YuNetDetector(BaseFaceDetector):
    def __init__(self, confidence_threshold: float = 0.6, model_path: str = None):
        super().__init__(confidence_threshold)
        
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = os.path.join(base_dir, "data", "models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
            
        self.model_path = model_path
        self._ensure_model_exists()
        
        # Initialize detector with default dummy size; updated dynamically per input frame
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=confidence_threshold,
            nms_threshold=0.3,
            top_k=5000
        )
        self.current_size = (320, 320)

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"[INFO] Downloading YuNet model to {self.model_path}...")
            try:
                urllib.request.urlretrieve(YUNET_MODEL_URL, self.model_path)
                print("[INFO] YuNet model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to download YuNet model from {YUNET_MODEL_URL}: {e}")

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []
            
        h, w = image.shape[:2]
        if (w, h) != self.current_size:
            self.detector.setInputSize((w, h))
            self.current_size = (w, h)
            
        _, faces = self.detector.detect(image)
        if faces is None:
            return []
            
        results = []
        for face in faces:
            # Face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
            x, y, bw, bh = map(int, face[0:4])
            score = float(face[-1])
            
            if score < self.confidence_threshold:
                continue
                
            # Extract 5 keypoints: right_eye, left_eye, nose_tip, right_mouth, left_mouth
            landmarks = []
            for i in range(5):
                kp_x = int(face[4 + i * 2])
                kp_y = int(face[5 + i * 2])
                landmarks.append((kp_x, kp_y))
                
            bbox = (max(0, x), max(0, y), int(bw), int(bh))
            crop = self.crop_face(image, bbox)
            
            results.append(DetectionResult(
                bbox=bbox,
                confidence=score,
                landmarks=landmarks,
                face_crop=crop
            ))
            
        return results
