"""
Abstract Base Class for Face Detectors
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None  # [(x,y), ...] for 5 points
    face_crop: Optional[np.ndarray] = None

class BaseFaceDetector(ABC):
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect faces in a BGR numpy image.
        Returns a list of DetectionResult objects.
        """
        pass

    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], margin_pct: float = 0.1) -> np.ndarray:
        """
        Safely crop a face from image given bbox with optional margin.
        """
        h, w = image.shape[:2]
        x, y, bw, bh = bbox
        
        # Add margin
        mx = int(bw * margin_pct)
        my = int(bh * margin_pct)
        
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + bw + mx)
        y2 = min(h, y + bh + my)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
        return crop
