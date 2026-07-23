"""
OpenCV Haar Cascade Face Detector
"""

import cv2
import numpy as np
from typing import List
from .base import BaseFaceDetector, DetectionResult

class HaarCascadeDetector(BaseFaceDetector):
    def __init__(self, confidence_threshold: float = 0.5):
        super().__init__(confidence_threshold)
        if hasattr(cv2, "CascadeClassifier"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            self.fallback = None
        else:
            print("[INFO] cv2.CascadeClassifier not present in this OpenCV build. Falling back to YuNet detector for HaarCascadeDetector request.")
            from .yunet import YuNetDetector
            self.fallback = YuNetDetector(confidence_threshold=confidence_threshold)

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []
            
        if self.fallback is not None:
            return self.fallback.detect(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        rects = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        results = []
        for (x, y, w, h) in rects:
            bbox = (int(x), int(y), int(w), int(h))
            crop = self.crop_face(image, bbox)
            results.append(DetectionResult(
                bbox=bbox,
                confidence=0.85,
                landmarks=None,
                face_crop=crop
            ))
        return results
