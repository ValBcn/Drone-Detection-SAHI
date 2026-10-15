# Advanced Drone Detection with SAHI and Adaptive Tracking

YOLOv8-based drone detection enhanced with SAHI (Slicing Aided Hyper Inference), temporal gradient processing, and adaptive confidence tracking for detecting small, fast-moving drones in complex environments.

## Features

### 🎯 SAHI Sliced Inference
- **Tile-based detection** for small objects
- Configurable slice size and overlap
- Detects drones that are <10 pixels in full frame

### 🔄 Temporal Gradient Enhancement
- Formula: `p2(t) = p(t) + lambda * (2*p(t) - p(t-offset) - p(t+offset))`
- Enhances moving objects while suppressing static background
- Improves detection when drone passes in front of trees

### 🎯 Adaptive Confidence Tracking
- **Progressive threshold lowering** around tracked drones
- Velocity-capped extrapolation (prevents false detections across image)
- Only searches near predicted positions with very low confidence
- Maintains tracking through occlusions

### 🎨 Channel-Specific Processing
- Remove green channel to improve contrast when drone is in front of trees
- Configurable color channel removal (red/green/blue)

### 📏 Size Filtering
- Filter bounding boxes by size relative to tile
- Removes large false positives (clouds, birds, buildings)

## Scripts

### Main Inference: `run_inference_sahi.py`
Full-featured SAHI inference with all enhancements:

```bash
python run_inference_sahi.py \
  --max-frames 300 \
  --conf 0.1 \
  --slice-size 512 \
  --overlap-ratio 0.2 \
  --max-box-size 0.2 \
  --temporal-gradient \
  --temporal-lambda 5 \
  --temporal-offset 5 \
  --adaptive-conf \
  --adaptive-conf-min 0.005 \
  --adaptive-conf-step 0.005 \
  --adaptive-max-dist 50 \
  --adaptive-max-vel 15 \
  --adaptive-max-lost 3 \
  --remove-channel green
```

**Parameters:**
- `--conf`: Base confidence threshold
- `--slice-size`: Tile size for SAHI (smaller = better for tiny drones)
- `--max-box-size`: Max box size as fraction of tile (e.g., 0.2 = 1/5)
- `--temporal-lambda`: Weight for temporal gradient (0 = disabled)
- `--adaptive-conf`: Enable adaptive confidence tracking
- `--adaptive-max-dist`: Max distance from predicted position (pixels)
- `--adaptive-max-vel`: Max velocity cap (pixels/frame)
- `--remove-channel`: Remove color channel (green/red/blue/none)

### Basic Inference: `run_inference.py`
Standard YOLOv8 inference with ByteTrack tracking:

```bash
python run_inference.py --track --max-box-size 0.2
```

### Video Inspector: `inspect_video_channels.py`
Manual frame-by-frame channel inspection:

```python
# Edit variables at top of script:
frame_number = 150
zoom_x = 1500
zoom_y = 500
zoom_size = 200

# Run
python inspect_video_channels.py
```

## Setup

### Environment
```bash
conda create -p /path/to/conda/envs/drone_detection python=3.10
conda activate /path/to/conda/envs/drone_detection
pip install ultralytics sahi opencv-python pillow matplotlib pandas
```

### Model Weights
Download pre-trained YOLOv8x weights from [HuggingFace](https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x/tree/main/weight):
```bash
mkdir -p weights
cd weights
wget https://huggingface.co/doguilmak/Drone-Detection-YOLOv8x/resolve/main/weight/best.pt
```

## Use Cases

**Small drones in surveillance footage:**
```bash
python run_inference_sahi.py --conf 0.1 --slice-size 416 --max-box-size 0.15 --adaptive-conf
```

**Drone passing in front of trees:**
```bash
python run_inference_sahi.py --remove-channel green --temporal-gradient --temporal-lambda 5 --adaptive-conf
```

**Fast-moving drones:**
```bash
python run_inference_sahi.py --adaptive-max-vel 30 --adaptive-max-dist 80 --track
```

## How It Works

### SAHI (Slicing Aided Hyper Inference)
- Splits each frame into overlapping tiles (e.g., 512×512)
- Runs detection on each tile independently
- Merges results, removing duplicates at tile boundaries
- A 5-pixel drone in full frame becomes 30+ pixels in a tile

### Temporal Gradient
- Enhances pixels that change between past and future frames
- Moving drone gets bright, static background gets dark
- Formula uses symmetric difference: `2*p(t) - p(t-5) - p(t+5)`

### Adaptive Confidence
1. Drone detected at conf 0.1 → track established
2. Next frame: try conf 0.1 first
3. If not found near predicted position:
   - Try 0.09, then 0.08, then 0.07...
   - Stop when at least one detection found
   - Only searches within 50px of predicted position
4. Velocity capped at 20px/frame to prevent wild extrapolation

## Original Work

Based on [Drone-Detection-YOLOv8x](https://github.com/doguilmak/Drone-Detection-YOLOv8x) by [@doguilmak](https://github.com/doguilmak).

**Dataset:** [YOLO Drone Detection Dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset)

**Related projects:**
- [Drone-Detection-YOLOv11x](https://github.com/doguilmak/Drone-Detection-YOLOv11x)
- [Drone-Detection-YOLOv7](https://github.com/doguilmak/Drone-Detection-YOLOv7)

## Output

- Detection boxes: **Red** for normal detections, **Yellow** for adaptive detections
- Yellow circles show predicted drone positions
- Yellow rectangles show adaptive search regions
- Confidence scores displayed on each detection
- Output video: `output/<video_name>_detected_sahi.mp4`

## License

This project extends the original Drone-Detection-YOLOv8x work with additional features for small object detection and tracking.
