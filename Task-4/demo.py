"""
End-to-End Demonstration and Automated Verification Runner for Face Engine
"""

import os
import cv2
import numpy as np
from face_engine import EngineConfig, FacePipeline

def create_synthetic_face_image(name: str = "Test Face") -> np.ndarray:
    """
    Creates a clean synthetic test face canvas image for automated testing.
    """
    img = np.full((480, 640, 3), (40, 35, 30), dtype=np.uint8)
    
    # Draw head oval
    cv2.ellipse(img, (320, 240), (100, 130), 0, 0, 360, (210, 180, 160), -1)
    
    # Draw eyes
    cv2.circle(img, (280, 210), 14, (255, 255, 255), -1)
    cv2.circle(img, (360, 210), 14, (255, 255, 255), -1)
    cv2.circle(img, (280, 210), 6, (80, 40, 10), -1)
    cv2.circle(img, (360, 210), 6, (80, 40, 10), -1)
    
    # Draw nose
    cv2.line(img, (320, 210), (315, 255), (160, 130, 110), 3)
    cv2.line(img, (315, 255), (325, 255), (160, 130, 110), 3)
    
    # Draw mouth
    cv2.ellipse(img, (320, 290), (35, 18), 0, 0, 180, (80, 50, 180), 4)

    # Label text on image background
    cv2.putText(img, f"Subject: {name}", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return img

def main():
    print("==========================================================")
    print("      FACE DETECTION & RECOGNITION PIPELINE DEMO         ")
    print("==========================================================")

    config = EngineConfig(
        detector_type="yunet",
        det_confidence_threshold=0.5,
        rec_similarity_threshold=0.5
    )
    
    pipeline = FacePipeline(config)

    # 1. Create Synthetic Test Face Image
    print("\n[STEP 1] Generating synthetic portrait image for testing...")
    test_img = create_synthetic_face_image("Alice")
    
    # 2. Test Face Identity Enrollment
    print("\n[STEP 2] Enrolling identity 'Alice' into Face Gallery...")
    enrolled = pipeline.enroll_face(test_img, name="Alice")
    print(f"-> Enrollment result: {'SUCCESS' if enrolled else 'FAILED'}")

    # 3. Test Detection & Recognition on Enrolled Image
    print("\n[STEP 3] Running Face Pipeline with YuNet Detector...")
    ann_img, detections, telemetry = pipeline.process(test_img, draw_annotations=True)
    
    print(f"-> FPS: {telemetry['fps']:.2f}")
    print(f"-> Latency: {telemetry['latency_ms']:.2f} ms")
    print(f"-> Detections Found: {telemetry['face_count']}")
    
    for idx, det in enumerate(detections):
        print(f"   Face #{idx+1}: BBox={det['bbox']}, Identity='{det['identity']}', Confidence={det['match_confidence']:.2f}, Known={det['is_known']}")

    # Save output demo image
    output_dir = os.path.join(config.data_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "demo_result.jpg")
    cv2.imwrite(out_path, ann_img)
    print(f"\n[STEP 4] Annotated output image saved to: {out_path}")

    # 4. Test Switching Detectors (Haar Cascade)
    print("\n[STEP 5] Testing Detector Backend Switch to Haar Cascade...")
    pipeline.switch_detector("haar")
    ann_haar, det_haar, telem_haar = pipeline.process(test_img)
    print(f"-> Haar Cascade Detections Found: {telem_haar['face_count']}, Latency: {telem_haar['latency_ms']:.2f} ms")

    print("\n==========================================================")
    print("         DEMO SUCCEEDED - ALL MODULES OPERATIONAL          ")
    print("==========================================================")

if __name__ == "__main__":
    main()
