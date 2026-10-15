#!/usr/bin/env python3
"""
Drone Detection Inference Script
Runs YOLOv8x inference on video and outputs annotated video with bounding boxes and confidence scores.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Drone Detection with YOLOv8x')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process (for testing)')
    parser.add_argument('--conf', type=float, default=0.05,
                        help='Confidence threshold (default: 0.05)')
    parser.add_argument('--iou', type=float, default=0.3,
                        help='IoU threshold for NMS (default: 0.3)')
    parser.add_argument('--max-box-size', type=float, default=None,
                        help='Maximum bounding box size as fraction of image (e.g., 0.2 for 1/5)')
    parser.add_argument('--track', action='store_true',
                        help='Enable ByteTrack tracking for spatial consistency')
    args = parser.parse_args()
    # Paths
    weights_path = "weights/best.pt"
    video_path = "data/6aad08c49813a531bca39a0a_IPCAM_ll1ttkgo.mp4"
    output_dir = "output"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get video name for output
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{video_name}_detected.mp4")

    print("=" * 60)
    print("Drone Detection Inference")
    print("=" * 60)
    print(f"Model weights: {weights_path}")
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")
    print(f"Max frames: {args.max_frames if args.max_frames else 'All'}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Max box size: {args.max_box_size if args.max_box_size else 'No limit'}")
    print(f"Tracking: {'ByteTrack' if args.track else 'Disabled'}")
    print("=" * 60)

    # Load the trained model
    print("\nLoading YOLOv8x model with trained weights...")
    model = YOLO(weights_path)
    print("Model loaded successfully!")

    # Run inference on the video
    print("\nRunning inference on video...")
    print("This may take a while depending on video length...\n")

    # Handle frame limiting if specified
    source = video_path
    temp_video = None

    if args.max_frames:
        print(f"Limiting to first {args.max_frames} frames...")
        temp_video = f"{output_dir}/temp_input.mp4"

        # Extract first N frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        frame_count = 0
        while frame_count < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        source = temp_video
        print(f"Extracted {frame_count} frames to temporary video")

    # Run detection/tracking
    if args.max_box_size:
        # Custom processing with size filtering
        print(f"Filtering boxes larger than {args.max_box_size} of image size...")

        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prepare output
        os.makedirs(f"{output_dir}/detection", exist_ok=True)
        output_avi = f"{output_dir}/detection/{video_name}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_writer = cv2.VideoWriter(output_avi, fourcc, fps, (width, height))

        frame_num = 0
        total_detections = 0
        filtered_detections = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            results = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)

            # Filter and draw boxes
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()

                    total_detections += 1

                    # Calculate box size relative to image
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height

                    # Filter by size
                    if box_width <= args.max_box_size and box_height <= args.max_box_size:
                        filtered_detections += 1
                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # Draw label with confidence
                        label = f"drone {conf:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out_writer.write(frame)
            frame_num += 1
            if frame_num % 100 == 0:
                print(f"Processed {frame_num} frames...")

        cap.release()
        out_writer.release()

        print(f"\nTotal detections: {total_detections}")
        print(f"Kept after filtering: {filtered_detections}")
        print(f"Filtered out: {total_detections - filtered_detections}")

    else:
        # Standard YOLO processing
        if args.track:
            # Use built-in YOLO tracking with ByteTrack
            print("Using YOLO's built-in ByteTrack tracker...")
            results = model.track(
                source=source,
                conf=args.conf,
                iou=args.iou,
                tracker="bytetrack.yaml",  # Built-in ByteTrack tracker
                persist=True,              # Persist tracks between frames
                save=True,
                show_labels=True,
                show_conf=True,
                line_width=2,
                project=output_dir,
                name="detection",
                exist_ok=True
            )
        else:
            # Frame-by-frame prediction without tracking
            results = model.predict(
                source=source,
                conf=args.conf,
                iou=args.iou,
                save=True,
                show_labels=True,
                show_conf=True,
                line_width=2,
                project=output_dir,
                name="detection",
                exist_ok=True
            )

    # Clean up temp video if created
    if temp_video and os.path.exists(temp_video):
        os.remove(temp_video)

    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)

    # Convert to compressed MP4
    print("\nConverting to MP4...")

    # Determine AVI location based on processing mode
    if args.max_box_size:
        avi_file = f"{output_dir}/detection/{video_name}.avi"
    else:
        # YOLO saves to runs/detect/output/detection by default
        avi_file = f"runs/detect/{output_dir}/detection/{video_name}.avi"

    mp4_file = f"{output_dir}/{video_name}_detected.mp4"

    print(f"Looking for AVI at: {avi_file}")

    if os.path.exists(avi_file):
        result = subprocess.run([
            "ffmpeg", "-y", "-i", avi_file,
            "-c:v", "libx264", "-crf", "23", "-preset", "medium",
            mp4_file
        ], capture_output=True, text=True)

        if result.returncode == 0:
            # Get file sizes
            avi_size = os.path.getsize(avi_file) / (1024 * 1024)  # MB
            mp4_size = os.path.getsize(mp4_file) / (1024 * 1024)  # MB

            print(f"AVI size: {avi_size:.1f} MB")
            print(f"MP4 size: {mp4_size:.1f} MB")
            print(f"Saved {(avi_size - mp4_size):.1f} MB ({(1 - mp4_size/avi_size)*100:.1f}% reduction)")

            # Remove the AVI file
            os.remove(avi_file)
            print(f"\nOutput video saved to: {mp4_file}")
        else:
            print(f"Error converting video: {result.stderr}")
            mp4_file = avi_file
    else:
        print(f"Error: AVI file not found at {avi_file}")
        print("Searching for output file...")
        # Search for any avi file in runs directory
        import glob
        found = glob.glob("runs/detect/**/detection/*.avi", recursive=True)
        if found:
            print(f"Found: {found[0]}")
            mp4_file = found[0]
        else:
            mp4_file = "NOT FOUND"

    # Print summary
    print("\nProcessing Summary:")
    print(f"- Input: {video_path}")
    print(f"- Output: {mp4_file}")
    print(f"- Frames processed: {args.max_frames if args.max_frames else 'All'}")
    print(f"- Model: YOLOv8x (Drone Detection)")
    print(f"- Confidence threshold: {args.conf}")
    print(f"- IoU threshold: {args.iou}")
    print(f"- Max box size: {args.max_box_size if args.max_box_size else 'No limit'}")
    print(f"- Tracking: {'ByteTrack' if args.track else 'Disabled'}")

if __name__ == "__main__":
    main()
