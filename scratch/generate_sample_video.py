"""
Utility script to generate a sample dashcam video from test set frames.
Sequences normal road driving, pothole distress passage, and normal road recovery.
"""

from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sample_videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_VIDEO_PATH = OUTPUT_DIR / "sample_dashcam.mp4"

TEST_NORMAL_DIR = PROJECT_ROOT / "dataset" / "test" / "Normal"
TEST_POTHOLE_DIR = PROJECT_ROOT / "dataset" / "test" / "Pothole"

normal_images = sorted(list(TEST_NORMAL_DIR.glob("*.jpg")))
pothole_images = sorted(list(TEST_POTHOLE_DIR.glob("*.jpg")))

print(f"Found {len(normal_images)} normal test images and {len(pothole_images)} pothole test images.")

if not normal_images or not pothole_images:
    print("Test images not found.")
    exit(0)

# Build a simulated video sequence:
# 1. 20 frames normal
# 2. 15 frames pothole hazard zone
# 3. 20 frames normal road
sequence_files = []
for i in range(min(15, len(normal_images))):
    sequence_files.extend([normal_images[i]] * 3)

for j in range(min(10, len(pothole_images))):
    sequence_files.extend([pothole_images[j]] * 3)

for k in range(min(15, len(normal_images))):
    sequence_files.extend([normal_images[-(k+1)]] * 3)

print(f"Constructing video with {len(sequence_files)} frames...")

target_w, target_h = 640, 480
fps = 15.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_VIDEO_PATH), fourcc, fps, (target_w, target_h))

for fpath in sequence_files:
    img = cv2.imread(str(fpath))
    if img is None:
        continue
    resized = cv2.resize(img, (target_w, target_h))
    writer.write(resized)

writer.release()
print(f"Sample video created successfully at: {OUT_VIDEO_PATH} ({len(sequence_files)} frames, {len(sequence_files)/fps:.1f}s)")
