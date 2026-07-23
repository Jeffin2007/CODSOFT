"""
Modern Computer Vision Visualization & Rendering Engine
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ..config import EngineConfig

class Visualizer:
    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()

    def draw_rounded_rectangle(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                              color: Tuple[int, int, int], thickness: int = 2, r: int = 12):
        """
        Draw sleek bounding box with rounded corners.
        """
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Ensure dimensions are sufficient for radius r
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        r = min(r, w // 4, h // 4)
        
        if r <= 2:
            cv2.rectangle(img, pt1, pt2, color, thickness)
            return

        # Top-left corner
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        # Top-right corner
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        # Bottom-right corner
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        # Bottom-left corner
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)

        # Connect straight lines
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)

    def draw_detection(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        identity_name: str = "Unknown",
        confidence: float = 0.0,
        det_confidence: float = 0.0,
        landmarks: Optional[List[Tuple[int, int]]] = None,
        is_known: bool = False
    ) -> np.ndarray:
        """
        Annotate image with sleek face bounding box, identity badge, landmarks, and score metrics.
        """
        output = image.copy()
        x, y, w, h = bbox
        pt1 = (x, y)
        pt2 = (x + w, y + h)

        color = self.config.color_known if is_known else self.config.color_unknown

        # 1. Semi-transparent box glow/tint
        overlay = output.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, 0.08, output, 0.92, 0, output)

        # 2. Draw rounded outer bounding box
        self.draw_rounded_rectangle(output, pt1, pt2, color=color, thickness=2, r=10)

        # 3. Draw facial landmarks (if available)
        if self.config.draw_landmarks and landmarks:
            for idx, (lx, ly) in enumerate(landmarks):
                # Distinct colors for eyes, nose, mouth
                if idx in [0, 1]:  # eyes
                    kp_color = (255, 220, 50)
                elif idx == 2:      # nose
                    kp_color = (50, 255, 255)
                else:               # mouth corners
                    kp_color = (255, 100, 255)
                cv2.circle(output, (lx, ly), 3, kp_color, -1, cv2.LINE_AA)

        # 4. Draw Identity Badge Tag above bounding box
        label_text = f"{identity_name}"
        if self.config.draw_confidence:
            if is_known:
                label_text += f" ({int(confidence * 100)}%)"
            else:
                label_text += f" [Det: {int(det_confidence * 100)}%]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1
        
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        badge_y1 = max(0, y - text_h - 14)
        badge_y2 = y
        badge_x1 = x
        badge_x2 = x + text_w + 16

        # Badge background
        cv2.rectangle(output, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        
        # Text inside badge
        cv2.putText(
            output,
            label_text,
            (badge_x1 + 8, badge_y2 - 6),
            font,
            font_scale,
            (0, 0, 0) if is_known else (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

        return output

    def draw_hud(
        self,
        image: np.ndarray,
        fps: float,
        face_count: int,
        detector_name: str,
        latency_ms: float
    ) -> np.ndarray:
        """
        Renders performance telemetry HUD banner on top of the image.
        """
        output = image.copy()
        if not self.config.draw_fps:
            return output

        h, w = output.shape[:2]
        hud_h = 36
        
        # Transparent top bar
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 20, 30), -1)
        cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)

        # HUD status items
        items = [
            f"FPS: {fps:.1f}",
            f"Faces: {face_count}",
            f"Engine: {detector_name.upper()}",
            f"Latency: {latency_ms:.1f} ms"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        curr_x = 15
        for item in items:
            cv2.putText(output, item, (curr_x, 23), font, font_scale, (230, 240, 255), thickness, cv2.LINE_AA)
            curr_x += 160

        return output
