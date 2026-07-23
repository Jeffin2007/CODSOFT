"""
Centralized Configuration for Face Engine
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import os

@dataclass
class EngineConfig:
    # Model choices
    detector_type: str = "yunet"  # Options: 'yunet', 'haar', 'mtcnn'
    embedder_type: str = "facenet"  # Options: 'facenet'
    
    # Thresholds
    det_confidence_threshold: float = 0.6
    rec_similarity_threshold: float = 0.60
    
    # Execution options
    device: str = "cpu"  # 'cpu' or 'cuda'
    draw_landmarks: bool = True
    draw_confidence: bool = True
    draw_fps: bool = True
    
    # Image & Face Preprocessing
    face_crop_size: Tuple[int, int] = (160, 160)
    input_image_max_dim: int = 1280
    
    # Paths
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    gallery_db_path: str = field(init=False)
    
    # Visual aesthetics (BGR format for OpenCV)
    color_known: Tuple[int, int, int] = (62, 214, 96)     # Radiant Green
    color_unknown: Tuple[int, int, int] = (43, 90, 255)   # Coral Orange/Red
    color_hud: Tuple[int, int, int] = (245, 175, 50)      # Electric Cyan/Blue
    color_text: Tuple[int, int, int] = (255, 255, 255)    # Pure White

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.gallery_db_path = os.path.join(self.data_dir, "identities.json")
