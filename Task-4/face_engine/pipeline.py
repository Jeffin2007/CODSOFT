"""
Unified Face Detection & Recognition Pipeline
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from .config import EngineConfig
from .detectors import HaarCascadeDetector, YuNetDetector, MTCNNDetector, BaseFaceDetector
from .recognizers import FaceNetEmbedder, FaceGallery, BaseFaceEmbedder
from .utils import Visualizer

class FacePipeline:
    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()
        self.visualizer = Visualizer(self.config)
        self.gallery = FaceGallery(self.config.gallery_db_path)
        
        # Initialize Embedder
        self.embedder: BaseFaceEmbedder = FaceNetEmbedder(device=self.config.device)
        
        # Initialize Detector
        self.detector: Optional[BaseFaceDetector] = None
        self.switch_detector(self.config.detector_type)

    def switch_detector(self, detector_type: str):
        """
        Dynamically switch active detection backend.
        """
        detector_type = detector_type.lower()
        print(f"[INFO] Switching face detector to: {detector_type}")
        if detector_type == "yunet":
            self.detector = YuNetDetector(
                confidence_threshold=self.config.det_confidence_threshold
            )
        elif detector_type == "haar":
            self.detector = HaarCascadeDetector(
                confidence_threshold=self.config.det_confidence_threshold
            )
        elif detector_type == "mtcnn":
            self.detector = MTCNNDetector(
                confidence_threshold=self.config.det_confidence_threshold,
                device=self.config.device
            )
        else:
            raise ValueError(f"Unsupported detector type: '{detector_type}'. Choose 'yunet', 'haar', or 'mtcnn'.")
            
        self.config.detector_type = detector_type

    def enroll_face(self, image: np.ndarray, name: str, bbox: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Detect face (or use provided bbox) from image, extract embedding, and enroll into gallery.
        """
        if image is None or image.size == 0 or not name.strip():
            return False

        if bbox is None:
            detections = self.detector.detect(image)
            if not detections:
                print(f"[WARNING] No face detected for enrollment of '{name}'.")
                return False
            # Select largest face
            detections.sort(key=lambda d: d.bbox[2] * d.bbox[3], reverse=True)
            crop = detections[0].face_crop
        else:
            crop = self.detector.crop_face(image, bbox)

        embedding = self.embedder.extract_embedding(crop)
        success = self.gallery.enroll(name.strip(), embedding)
        if success:
            print(f"[INFO] Enrolled face identity: '{name}' into gallery.")
        return success

    def process(self, image: np.ndarray, draw_annotations: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute end-to-end pipeline: Detect -> Embed -> Match Identity -> Render Visual Overlay.
        Returns:
            - annotated_image (np.ndarray)
            - detections_info (List[Dict])
            - telemetry (Dict)
        """
        start_time = time.time()
        
        if image is None or image.size == 0:
            return image, [], {"fps": 0.0, "latency_ms": 0.0, "face_count": 0}

        # 1. Detect faces
        detections = self.detector.detect(image)
        
        annotated_img = image.copy()
        detections_info = []

        # 2. Process each detected face
        for det in detections:
            crop = det.face_crop if det.face_crop is not None else self.detector.crop_face(image, det.bbox)
            
            # Extract embedding & match identity
            embedding = self.embedder.extract_embedding(crop)
            match = self.gallery.match(embedding, threshold=self.config.rec_similarity_threshold)

            det_dict = {
                "bbox": det.bbox,
                "confidence": det.confidence,
                "landmarks": det.landmarks,
                "identity": match.name,
                "match_confidence": match.confidence,
                "is_known": match.is_known
            }
            detections_info.append(det_dict)

            # Draw visual annotation
            if draw_annotations:
                annotated_img = self.visualizer.draw_detection(
                    image=annotated_img,
                    bbox=det.bbox,
                    identity_name=match.name,
                    confidence=match.confidence,
                    det_confidence=det.confidence,
                    landmarks=det.landmarks,
                    is_known=match.is_known
                )

        latency_ms = (time.time() - start_time) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        telemetry = {
            "fps": fps,
            "latency_ms": latency_ms,
            "face_count": len(detections_info),
            "detector": self.config.detector_type
        }

        # Render telemetry HUD
        if draw_annotations:
            annotated_img = self.visualizer.draw_hud(
                image=annotated_img,
                fps=fps,
                face_count=len(detections_info),
                detector_name=self.config.detector_type,
                latency_ms=latency_ms
            )

        return annotated_img, detections_info, telemetry
