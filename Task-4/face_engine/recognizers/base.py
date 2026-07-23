"""
Abstract Base Interface for Face Feature Embedding Models
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseFaceEmbedder(ABC):
    @abstractmethod
    def extract_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Takes a BGR face crop image and returns a 1D L2-normalized float feature vector (e.g. 512-dim).
        """
        pass
