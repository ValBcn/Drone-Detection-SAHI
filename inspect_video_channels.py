#!/usr/bin/env python3
"""
Simple Video Channel Inspector
Just 3 subplots (R, G, B) for manual inspection
"""

import cv2
import matplotlib.pyplot as plt

# Configuration - CHANGE THESE VALUES
video_path = "data/6aad08c49813a531bca39a0a_IPCAM_ll1ttkgo.mp4"

for i in range(150):

    frame_number = 150+i  # Which frame to display
    zoom_x = 1500      # X coordinate for zoom center
    zoom_y = 0      # Y coordinate for zoom center
    zoom_size = 1000   # Size of zoom region (pixels)
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error reading frame {frame_number}")
        exit()
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Extract ROI
    x1 = max(0, zoom_x - zoom_size // 2)
    y1 = max(0, zoom_y - zoom_size // 2)
    x2 = min(frame_rgb.shape[1], zoom_x + zoom_size // 2)
    y2 = min(frame_rgb.shape[0], zoom_y + zoom_size // 2)
    
    roi = frame_rgb[y1:y2, x1:x2]
    red_channel = roi[:, :, 0]
    green_channel = roi[:, :, 1]
    blue_channel = roi[:, :, 2]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display channels
    ax1.imshow(red_channel, cmap='Reds', vmin=0, vmax=255)
    ax1.set_title(f'Red Channel\nMean: {red_channel.mean():.1f}')
    ax1.axis('off')
    
    ax2.imshow(green_channel, cmap='Greens', vmin=0, vmax=255)
    ax2.set_title(f'Green Channel\nMean: {green_channel.mean():.1f}')
    ax2.axis('off')
    
    ax3.imshow(blue_channel, cmap='Blues', vmin=0, vmax=255)
    ax3.set_title(f'Blue Channel\nMean: {blue_channel.mean():.1f}')
    ax3.axis('off')
    
    fig.suptitle(f'Frame {frame_number} - ROI at ({zoom_x}, {zoom_y}) size {zoom_size}x{zoom_size}')
    plt.tight_layout()
    
    # Print statistics
    print(f"\nFrame {frame_number} - ROI Statistics:")
    print(f"Red   - Mean: {red_channel.mean():.2f}, Std: {red_channel.std():.2f}, Min: {red_channel.min()}, Max: {red_channel.max()}")
    print(f"Green - Mean: {green_channel.mean():.2f}, Std: {green_channel.std():.2f}, Min: {green_channel.min()}, Max: {green_channel.max()}")
    print(f"Blue  - Mean: {blue_channel.mean():.2f}, Std: {blue_channel.std():.2f}, Min: {blue_channel.min()}, Max: {blue_channel.max()}")
    
    plt.show()
