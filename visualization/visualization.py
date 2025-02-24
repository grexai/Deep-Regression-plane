import os
import json
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
from glob import glob
from PIL import Image

# Base directory
base_dir = r"d:\dev\DVP2\2022_v1_zeroPadded_split_with_test"
base_dir = "/storage01/grexai/datasets/Regplane_data/2022_v1_zeroPadded_split_with_test"

# Choose dataset split (train, val, test)
split = "test"  # Change to "train" or "test" if needed

# Paths
image_dir = os.path.join(base_dir, split, "images")
label2_dir = os.path.join(base_dir, split, "labels2")

# Load all JSON label files
json_files = glob(os.path.join(label2_dir, "*.json"))

# Create a blank canvas (white)
canvas_size = (10000, 10000, 3)  # 10000x10000 with 3 color channels (RGB)
canvas = np.ones(canvas_size, dtype=np.uint8) * 255  # White background

# Visualization parameters
thumb_size = 299  # Thumbnail size for each image
scatter_color = (255, 0, 0)  # Red dots for missing images

for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)

    x, y = data["x"], data["y"]
    img_name = os.path.basename(json_file).replace(".json", ".jpg")  # Assuming .jpg images

    img_path = os.path.join(image_dir, img_name)

    if not os.path.exists(img_path):
        img_name = img_name.replace(".jpg", ".png")  # Try .png if .jpg doesn't exist
        img_path = os.path.join(image_dir, img_name)

    if os.path.exists(img_path):
        # Load image and resize to thumbnail
        img = Image.open(img_path)
        img.thumbnail((thumb_size, thumb_size))

        # Convert to NumPy array and overlay on the canvas
        img_array = np.array(img)
        h, w, _ = img_array.shape

        # Ensure coordinates stay within bounds
        x, y = min(x, 9900), min(y, 9900)  # Avoid going outside 10000x10000
        canvas[y:y + h, x:x + w] = img_array
    else:
        # If image is missing, draw a red dot at (x, y)
        cv2.circle(canvas, (x, y), 10, scatter_color, -1)

# Convert canvas to PIL image and display
plt.figure(figsize=(8, 8))
plt.tight_layout()
plt.imshow(canvas)
plt.axis("off")
plt.title(f"{split.upper()} Data Visualization")
plt.savefig("data visualization.jpg")
