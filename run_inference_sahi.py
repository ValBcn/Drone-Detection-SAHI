#!/usr/bin/env python3
"""
Drone detection with YOLOv8 + SAHI (sliced inference).

Optional adaptive search (--adaptive-conf): detections are tracked from frame to frame.
When a track is missed, the region around its predicted position is searched again
with a lower confidence threshold.
"""

import os
import argparse
import subprocess
from pathlib import Path

import numpy as np
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def parse_args():
    p = argparse.ArgumentParser(description='Drone detection with YOLOv8 + SAHI')
    p.add_argument('--video', default='output/6aad08c49813a531bca39a0a_IPCAM_ll1ttkgo_median_diff.mp4')
    p.add_argument('--weights', default='weights/best.pt')
    p.add_argument('--output-dir', default='output')
    p.add_argument('--device', default='cuda:0', help="'cuda:0' or 'cpu'")
    p.add_argument('--max-frames', type=int, default=None, help='Stop after N frames')

    p.add_argument('--conf', type=float, default=0.2, help='Confidence threshold (default: 0.2)')
    p.add_argument('--slice-size', type=int, default=512, help='SAHI slice size in pixels (default: 512)')
    p.add_argument('--overlap-ratio', type=float, default=0.2, help='SAHI slice overlap (default: 0.2)')
    p.add_argument('--max-box-size', type=float, default=0.2,
                   help='Drop boxes larger than this fraction of the slice, 0 to disable (default: 0.2)')

    p.add_argument('--adaptive-conf', action='store_true',
                   help='Track detections and search again around missed tracks with a lower threshold')
    p.add_argument('--adaptive-conf-min', type=float, default=0.001,
                   help='Lowest confidence used in the search (default: 0.001)')
    p.add_argument('--adaptive-conf-step', type=float, default=0.05,
                   help='Step for lowering the confidence (default: 0.05)')
    p.add_argument('--adaptive-max-dist', type=int, default=50,
                   help='Max distance in pixels between a detection and the predicted position, also '
                        'half the size of the search region (default: 50)')
    p.add_argument('--adaptive-max-lost', type=int, default=10,
                   help='Frames a track is kept without being detected (default: 10)')
    p.add_argument('--adaptive-max-vel', type=float, default=1,
                   help='Max velocity in pixels per frame used for prediction (default: 1)')
    return p.parse_args()


def predict_position(track, frame_num, max_vel):
    """Last position + velocity (capped) * frames elapsed since last detection."""
    (x, y), f = track['positions'][-1], track['frames'][-1]
    if len(track['positions']) < 2:
        return x, y
    (px, py), pf = track['positions'][-2], track['frames'][-2]
    vx = np.clip((x - px) / (f - pf), -max_vel, max_vel)
    vy = np.clip((y - py) / (f - pf), -max_vel, max_vel)
    elapsed = frame_num - f
    return x + vx * elapsed, y + vy * elapsed


def to_detection(box, conf, adaptive=False):
    x1, y1, x2, y2 = box
    return {'box': box, 'centroid': ((x1 + x2) / 2, (y1 + y2) / 2), 'conf': conf, 'adaptive': adaptive}


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def main():
    args = parse_args()
    os.makedirs(f"{args.output_dir}/detection", exist_ok=True)
    video_name = Path(args.video).stem

    print(f"Video: {args.video}\nWeights: {args.weights}")
    print(f"conf={args.conf}, slice={args.slice_size}, overlap={args.overlap_ratio}, "
          f"max_box_size={args.max_box_size or 'off'}")
    if args.adaptive_conf:
        print(f"Adaptive search: conf_min={args.adaptive_conf_min}, step={args.adaptive_conf_step}, "
              f"max_dist={args.adaptive_max_dist}px, max_lost={args.adaptive_max_lost}, "
              f"max_vel={args.adaptive_max_vel}px/frame")
    

    def load_model(conf):
        return AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=args.weights,
                                                  confidence_threshold=conf, device=args.device)

    model = load_model(args.conf)
    adaptive_model = load_model(args.adaptive_conf_min) if args.adaptive_conf else None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_avi = f"{args.output_dir}/detection/{video_name}.avi"
    writer = cv2.VideoWriter(output_avi, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    def detect(image, det_model, slice_size, offset=(0, 0)):
        """Sliced prediction; returns [(box, conf)] in full-frame coordinates."""
        result = get_sliced_prediction(image, det_model, slice_height=slice_size, slice_width=slice_size,
                                       overlap_height_ratio=args.overlap_ratio,
                                       overlap_width_ratio=args.overlap_ratio, verbose=0)
        ox, oy = offset
        return [((int(p.bbox.minx) + ox, int(p.bbox.miny) + oy, int(p.bbox.maxx) + ox, int(p.bbox.maxy) + oy),
                 p.score.value) for p in result.object_prediction_list]

    def adaptive_search(frame, pred, detections):
        """Search around pred with progressively lower thresholds.
        Returns (detection or None, threshold or None, searched region or None)."""
        r = args.adaptive_max_dist
        rx1, ry1 = max(0, int(pred[0] - r)), max(0, int(pred[1] - r))
        rx2, ry2 = min(width, int(pred[0] + r)), min(height, int(pred[1] + r))
        if rx2 <= rx1 or ry2 <= ry1:
            return None, None, None
        region = (rx1, ry1, rx2, ry2)

        candidates = []
        for box, conf in detect(frame[ry1:ry2, rx1:rx2], adaptive_model, args.slice_size // 2, (rx1, ry1)):
            det = to_detection(box, conf, adaptive=True)
            if dist(det['centroid'], pred) >= args.adaptive_max_dist:
                continue
            if any(dist(det['centroid'], d['centroid']) < 20 for d in detections):
                continue  # duplicate of an existing detection
            candidates.append(det)

        # Highest threshold band that contains a candidate, then the closest one in it
        threshold = args.conf - args.adaptive_conf_step
        while candidates and threshold >= args.adaptive_conf_min:
            kept = [c for c in candidates if c['conf'] >= threshold]
            if kept:
                return min(kept, key=lambda c: dist(c['centroid'], pred)), threshold, region
            threshold -= args.adaptive_conf_step
        return None, None, region

    tracks = {}  # id -> {'positions': [(x, y)], 'frames': [frame_num]}
    next_id = 0
    n_detections = n_adaptive = 0
    frame_num = 0

    while args.max_frames is None or frame_num < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- Detection on the full frame ----
        detections = []
        for box, conf in detect(frame, model, args.slice_size):
            x1, y1, x2, y2 = box
            if args.max_box_size and max(x2 - x1, y2 - y1) > args.max_box_size * args.slice_size:
                continue
            detections.append(to_detection(box, conf))

        searched_regions = []
        if args.adaptive_conf:
            predictions = {tid: predict_position(t, frame_num, args.adaptive_max_vel) for tid, t in tracks.items()}

            # ---- Adaptive search around tracks with no detection near their prediction ----
            for tid, pred in predictions.items():
                if not (0 < pred[0] < width and 0 < pred[1] < height):
                    continue
                if any(dist(d['centroid'], pred) < args.adaptive_max_dist for d in detections):
                    continue
                best, threshold, region = adaptive_search(frame, pred, detections)
                if region:
                    searched_regions.append((pred, region))
                if best:
                    print(f"Frame {frame_num}: adaptive detection conf={best['conf']:.3f} "
                          f"(threshold {threshold:.3f}) {dist(best['centroid'], pred):.0f}px from prediction")
                    detections.append(best)
                    n_adaptive += 1

            # ---- Match detections to tracks (closest pairs first) ----
            pairs = sorted((dist(d['centroid'], predictions[tid]), tid, i)
                           for tid in tracks for i, d in enumerate(detections))
            used_tracks, used_dets = set(), set()
            for d_, tid, i in pairs:
                if d_ >= args.adaptive_max_dist:
                    break
                if tid in used_tracks or i in used_dets:
                    continue
                tracks[tid]['positions'] = tracks[tid]['positions'][-10:] + [detections[i]['centroid']]
                tracks[tid]['frames'] = tracks[tid]['frames'][-10:] + [frame_num]
                used_tracks.add(tid)
                used_dets.add(i)

            # New tracks for unmatched detections; unmatched tracks are kept until max_lost
            for i, d in enumerate(detections):
                if i not in used_dets:
                    tracks[next_id] = {'positions': [d['centroid']], 'frames': [frame_num]}
                    next_id += 1
            tracks = {tid: t for tid, t in tracks.items()
                      if frame_num - t['frames'][-1] <= args.adaptive_max_lost}

        # ---- Drawing ----
        out = frame.copy()
        for (px, py), (rx1, ry1, rx2, ry2) in searched_regions:  # cyan: prediction + search region
            cv2.circle(out, (int(px), int(py)), 5, (255, 255, 0), 2)
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)
        for d in detections:  # red: normal detection, yellow: adaptive detection
            x1, y1, x2, y2 = d['box']
            color = (0, 255, 255) if d['adaptive'] else (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            label = f"drone {d['conf']:.2f}" + (" (A)" if d['adaptive'] else "")
            cv2.putText(out, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        writer.write(out)

        n_detections += len(detections)
        frame_num += 1
        if frame_num % 10 == 0:
            print(f"Processed {frame_num} frames, detections: {n_detections}, adaptive: {n_adaptive}")

    cap.release()
    writer.release()

    # ---- Convert to MP4 ----
    mp4_file = f"{args.output_dir}/{video_name}_detected_sahi.mp4"
    conv = subprocess.run(["ffmpeg", "-y", "-i", output_avi, "-c:v", "libx264", "-crf", "23",
                           "-preset", "medium", mp4_file], capture_output=True, text=True)
    if conv.returncode == 0:
        os.remove(output_avi)
    else:
        print(f"MP4 conversion failed, keeping AVI:\n{conv.stderr}")
        mp4_file = output_avi

    print(f"\nDone: {frame_num} frames, {n_detections} detections ({n_adaptive} adaptive)")
    print(f"Output: {mp4_file}")


if __name__ == "__main__":
    main()