"""
FaceNet InceptionResnetV1 Embedding Model Implementation
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from .base import BaseFaceEmbedder

class FaceNetEmbedder(BaseFaceEmbedder):
    def __init__(self, device: str = "cpu"):
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        try:
            from facenet_pytorch import InceptionResnetV1
            print("[INFO] Loading FaceNet (InceptionResnetV1) model...")
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        except Exception as e:
            print(f"[WARNING] Could not load pre-trained VGGFace2 weights directly: {e}. Falling back to default initialization.")
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(classify=False).eval().to(self.device)

        # Standard preprocessing transform for FaceNet: 160x160 RGB normalized
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        if face_crop is None or face_crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        # Convert BGR to RGB
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Transform and add batch dimension
        tensor = self.preprocess(rgb_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(tensor).squeeze(0).cpu().numpy()
            
        # Ensure L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.astype(np.float32)
