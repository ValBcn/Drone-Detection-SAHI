#!/usr/bin/env python3
"""
Per-channel median background subtraction.

For each frame t and each color channel C (blue, green, red), the background is the
per-pixel median of that channel over past frames [t - gap - window, t - gap).
It is then subtracted, weighted, from the same channel:

    C' = C - w_C * median(C)

The output is a full color video that can be used directly as the input of the
detection code.

Usage example:
    python color_median_subtraction.py --video data/my_video.mp4 --weight 0.8
    python color_median_subtraction.py --weight 0.5 1.0 0.5     # R G B weights
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser(description='Per-channel median background subtraction')
    parser.add_argument('--video', type=str, default='data/6aad08c49813a531bca39a0a_IPCAM_ll1ttkgo.mp4')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (default: output/<video>_color_median.mp4)')
    parser.add_argument('--weight', type=float, nargs='+', default=[0.5],
                        help='Weight of the subtracted median: one value for all channels, '
                             'or three values in R G B order (default: 1.0)')
    parser.add_argument('--window', type=int, default=3,
                        help='Number of past frames used for the median (default: 15)')
    parser.add_argument('--gap', type=int, default=0,
                        help='Skip the most recent frames when building the median, so a slow drone '
                             'is not absorbed into the background (default: 5)')
    parser.add_argument('--step', type=int, default=1,
                        help='Use every Nth past frame in the median (default: 1)')
    parser.add_argument('--median-filter', type=int, choices=[0, 3, 5], default=3,
                        help='Spatial median filter window size (3 or 5) applied to the frames used for '
                             'the background, 0 to disable (default: 3)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')
    args = parser.parse_args()

    if len(args.weight) == 1:
        weights_rgb = args.weight * 3
    elif len(args.weight) == 3:
        weights_rgb = args.weight
    else:
        raise SystemExit("--weight takes 1 value or 3 values (R G B)")
    weights_bgr = np.array(weights_rgb[::-1], dtype=np.float32)  # OpenCV frames are BGR

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output = args.output or f"output/{Path(args.video).stem}_color_median.mp4"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    history = deque(maxlen=args.window * args.step + args.gap)

    print(f"Input: {args.video} ({width}x{height}, {fps:.1f} fps)")
    print(f"C' = C - w * median(C), weights R G B = {weights_rgb}, median of {args.window} frames "
          f"(every {args.step}), skipping the last {args.gap}, median filter {args.median_filter or 'off'}")
    print(f"Output: {output}")

    frame_num = 0
    while True:
        if args.max_frames and frame_num >= args.max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        # Past frames available for the median: drop the most recent `gap`, then take every `step`
        past = list(history)
        if args.gap > 0:
            past = past[:-args.gap] if len(past) > args.gap else []
        past = past[::-1][::args.step][::-1]

        if len(past) >= max(3, args.window // 2):
            # Median computed independently for each channel (axis 0 = time)
            background = np.median(np.stack(past, axis=0), axis=0).astype(np.float32)
            out_frame = frame.astype(np.float32) - weights_bgr * background
            out_frame = np.clip(out_frame, 0, 255).astype(np.uint8)
        else:
            out_frame = frame  # not enough history yet: frame left unchanged

        history.append(cv2.medianBlur(frame, args.median_filter) if args.median_filter > 0 else frame)
        writer.write(out_frame)

        frame_num += 1
        if frame_num % 50 == 0:
            print(f"Processed {frame_num} frames")

    cap.release()
    writer.release()
    print(f"\nDone: {frame_num} frames written to {output}")


if __name__ == "__main__":
    main()