"""
PyTorch MTCNN Face Detector Implementation
"""

import cv2
import numpy as np
import torch
from typing import List
from .base import BaseFaceDetector, DetectionResult

class MTCNNDetector(BaseFaceDetector):
    def __init__(self, confidence_threshold: float = 0.7, device: str = "cpu"):
        super().__init__(confidence_threshold)
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                keep_all=True,
                select_largest=False,
                min_face_size=30,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=self.device
            )
        except ImportError:
            raise RuntimeError("facenet-pytorch is required for MTCNNDetector. Install via pip install facenet-pytorch.")

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []

        # Convert BGR to RGB for MTCNN / PyTorch
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes, probs, points = self.mtcnn.detect(rgb_img, landmarks=True)
        if boxes is None or probs is None:
            return []

        results = []
        for i in range(len(boxes)):
            score = float(probs[i])
            if score < self.confidence_threshold:
                continue

            box = boxes[i]
            x1, y1, x2, y2 = map(int, box)
            bw = x2 - x1
            bh = y2 - y1
            bbox = (max(0, x1), max(0, y1), int(bw), int(bh))

            landmarks = None
            if points is not None and i < len(points):
                pts = points[i]
                landmarks = [(int(p[0]), int(p[1])) for p in pts]

            crop = self.crop_face(image, bbox)
            results.append(DetectionResult(
                bbox=bbox,
                confidence=score,
                landmarks=landmarks,
                face_crop=crop
            ))

        return results
