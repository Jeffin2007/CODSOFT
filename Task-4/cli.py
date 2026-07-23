"""
Command Line Interface (CLI) for Face Detection and Recognition Engine
"""

import argparse
import sys
import os
import cv2
from face_engine import EngineConfig, FacePipeline
from face_engine.utils import MediaHandler

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production-Ready Computer Vision Face Detection & Recognition Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands available")

    # Command: detect
    p_detect = subparsers.add_parser("detect", help="Detect & recognize faces in a static image")
    p_detect.add_argument("-i", "--input", required=True, help="Path to input image file")
    p_detect.add_argument("-o", "--output", default="output_detected.jpg", help="Path to save annotated output image")
    p_detect.add_argument("-m", "--model", default="yunet", choices=["yunet", "haar", "mtcnn"], help="Face detector model")
    p_detect.add_argument("-t", "--threshold", type=float, default=0.6, help="Detection confidence threshold")
    p_detect.add_argument("-s", "--sim-threshold", type=float, default=0.6, help="Recognition similarity threshold")

    # Command: process-video
    p_video = subparsers.add_parser("process-video", help="Process a video file and save output video")
    p_video.add_argument("-i", "--input", required=True, help="Path to input video file (.mp4, .avi)")
    p_video.add_argument("-o", "--output", default="output_video.mp4", help="Path to save processed video")
    p_video.add_argument("-m", "--model", default="yunet", choices=["yunet", "haar", "mtcnn"], help="Face detector model")

    # Command: enroll
    p_enroll = subparsers.add_parser("enroll", help="Enroll a new face identity into the gallery database")
    p_enroll.add_argument("-n", "--name", required=True, help="Name of the person/identity")
    p_enroll.add_argument("-i", "--image", required=True, help="Path to photo containing person's face")

    # Command: webcam
    p_webcam = subparsers.add_parser("webcam", help="Launch live webcam desktop window with real-time tracking")
    p_webcam.add_argument("-c", "--camera", type=int, default=0, help="Camera index")
    p_webcam.add_argument("-m", "--model", default="yunet", choices=["yunet", "haar", "mtcnn"], help="Face detector model")

    # Command: identities
    subparsers.add_parser("identities", help="List all enrolled face identities in the database")

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = EngineConfig(
        detector_type=getattr(args, "model", "yunet"),
        det_confidence_threshold=getattr(args, "threshold", 0.6),
        rec_similarity_threshold=getattr(args, "sim_threshold", 0.6)
    )
    
    pipeline = FacePipeline(config)

    if args.command == "detect":
        img = MediaHandler.load_image(args.input)
        if img is None:
            print(f"[ERROR] Could not load image file: {args.input}")
            sys.exit(1)

        ann_img, detections, telemetry = pipeline.process(img)
        cv2.imwrite(args.output, ann_img)
        print(f"[SUCCESS] Processed image in {telemetry['latency_ms']:.1f}ms ({telemetry['fps']:.1f} FPS)")
        print(f"[SUCCESS] Faces Detected: {len(detections)}")
        for idx, det in enumerate(detections):
            print(f"  Face #{idx+1}: Identity='{det['identity']}', Match Confidence={det['match_confidence']*100:.1f}%")
        print(f"[SUCCESS] Saved annotated image to: {args.output}")

    elif args.command == "process-video":
        def frame_cb(frame):
            ann_frame, det_list, telem = pipeline.process(frame)
            return ann_frame, {"face_count": len(det_list)}

        def prog_cb(pct):
            sys.stdout.write(f"\rProgress: {pct*100:.1f}%")
            sys.stdout.flush()

        print(f"[INFO] Processing video: {args.input} -> {args.output}")
        res = MediaHandler.process_video_file(args.input, args.output, frame_cb, prog_cb)
        print(f"\n[SUCCESS] Completed video processing. Total Frames: {res['total_frames']}, Total Faces: {res['total_faces_detected']}")

    elif args.command == "enroll":
        img = MediaHandler.load_image(args.image)
        if img is None:
            print(f"[ERROR] Could not load image file: {args.image}")
            sys.exit(1)

        success = pipeline.enroll_face(img, name=args.name)
        if success:
            print(f"[SUCCESS] Enrolled face identity: '{args.name}' successfully.")
        else:
            print(f"[ERROR] Failed to enroll face for '{args.name}'. Ensure a face is visible in the photo.")

    elif args.command == "identities":
        identities = pipeline.gallery.get_identities()
        print(f"Enrolled Identities ({len(identities)} total):")
        for idx, name in enumerate(identities, 1):
            sample_count = len(pipeline.gallery.gallery.get(name, []))
            print(f"  {idx}. {name} ({sample_count} sample embeddings enrolled)")

    elif args.command == "webcam":
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam index {args.camera}")
            sys.exit(1)

        print("[INFO] Starting live webcam tracking. Press 'q' or 'ESC' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ann_frame, _, _ = pipeline.process(frame)
            cv2.imshow("Face Detection & Recognition - Antigravity CV Engine", ann_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
