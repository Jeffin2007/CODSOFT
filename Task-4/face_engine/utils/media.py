"""
Media Loading, Decoding, Base64 Conversion & Video File Processing Helper
"""

import cv2
import base64
import numpy as np
from typing import Generator, Tuple, Optional, Callable

class MediaHandler:
    @staticmethod
    def load_image(source) -> Optional[np.ndarray]:
        """
        Loads image from file path or bytes buffer.
        """
        if isinstance(source, str):
            img = cv2.imread(source)
            return img
        elif isinstance(source, bytes):
            nparr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        elif isinstance(source, np.ndarray):
            return source
        return None

    @staticmethod
    def b64_to_cv2(b64_str: str) -> Optional[np.ndarray]:
        """
        Converts Base64 image string to OpenCV BGR numpy array.
        """
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            return MediaHandler.load_image(img_bytes)
        except Exception as e:
            print(f"[ERROR] Failed to decode Base64 image: {e}")
            return None

    @staticmethod
    def cv2_to_b64(image: np.ndarray, format: str = ".jpg") -> str:
        """
        Converts OpenCV BGR numpy array to Base64 encoded string.
        """
        _, buffer = cv2.imencode(format, image)
        b64_str = base64.b64encode(buffer).decode("utf-8")
        mime = "jpeg" if format.lower() in [".jpg", ".jpeg"] else "png"
        return f"data:image/{mime};base64,{b64_str}"

    @staticmethod
    def process_video_file(
        input_path: str,
        output_path: str,
        frame_callback: Callable[[np.ndarray], Tuple[np.ndarray, dict]],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> dict:
        """
        Processes video file frame by frame, executes frame_callback, saves output video.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video file: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_faces_detected = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            processed_frame, info = frame_callback(frame)
            out.write(processed_frame)

            frame_count += 1
            total_faces_detected += info.get("face_count", 0)

            if progress_callback and total_frames > 0:
                progress_callback(frame_count / total_frames)

        cap.release()
        out.release()

        return {
            "total_frames": frame_count,
            "fps": fps,
            "resolution": (width, height),
            "total_faces_detected": total_faces_detected,
            "output_path": output_path
        }
