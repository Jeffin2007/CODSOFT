"""
Face Identity Gallery & Vector Matching Database
"""

import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class IdentityMatch:
    name: str
    confidence: float
    is_known: bool

class FaceGallery:
    def __init__(self, db_path: str = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "identities.json")
            
        self.db_path = db_path
        # Storage structure: name -> list of normalized numpy float32 vectors
        self.gallery: Dict[str, List[np.ndarray]] = {}
        self.mean_embeddings: Dict[str, np.ndarray] = {}
        self.load()

    def enroll(self, name: str, embedding: np.ndarray) -> bool:
        """
        Enroll a new face embedding vector for a given identity name.
        """
        if embedding is None or len(embedding) == 0:
            return False

        # Ensure unit L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        if name not in self.gallery:
            self.gallery[name] = []

        self.gallery[name].append(embedding.astype(np.float32))
        self._update_mean(name)
        self.save()
        return True

    def remove_identity(self, name: str) -> bool:
        if name in self.gallery:
            del self.gallery[name]
            if name in self.mean_embeddings:
                del self.mean_embeddings[name]
            self.save()
            return True
        return False

    def get_identities(self) -> List[str]:
        return list(self.gallery.keys())

    def match(self, query_embedding: np.ndarray, threshold: float = 0.60) -> IdentityMatch:
        """
        Computes Cosine Similarity against all enrolled identities.
        Returns IdentityMatch object.
        """
        if query_embedding is None or len(self.mean_embeddings) == 0:
            return IdentityMatch(name="Unknown", confidence=0.0, is_known=False)

        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        best_name = "Unknown"
        best_sim = 0.0

        for name, mean_vec in self.mean_embeddings.items():
            # Cosine similarity for unit normalized vectors is dot product
            sim = float(np.dot(query_embedding, mean_vec))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        is_known = best_sim >= threshold and best_name != "Unknown"
        final_name = best_name if is_known else "Unknown"

        return IdentityMatch(
            name=final_name,
            confidence=max(0.0, min(1.0, best_sim)),
            is_known=is_known
        )

    def _update_mean(self, name: str):
        if name in self.gallery and self.gallery[name]:
            vectors = np.array(self.gallery[name])
            mean_vec = np.mean(vectors, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            self.mean_embeddings[name] = mean_vec.astype(np.float32)

    def save(self):
        """
        Save enrolled identities and embeddings to JSON.
        """
        serializable_data = {}
        for name, vec_list in self.gallery.items():
            serializable_data[name] = [vec.tolist() for vec in vec_list]

        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2)

    def load(self):
        """
        Load enrolled identities from JSON if available.
        """
        if not os.path.exists(self.db_path):
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.gallery.clear()
            self.mean_embeddings.clear()

            for name, vec_list in data.items():
                self.gallery[name] = [np.array(vec, dtype=np.float32) for vec in vec_list]
                self._update_mean(name)
        except Exception as e:
            print(f"[WARNING] Error loading gallery database from {self.db_path}: {e}")
