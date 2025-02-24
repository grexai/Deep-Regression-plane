import sys
import os
print(os.path.abspath('./')) 

sys.path.append(os.path.abspath('./'))

from regressionplane_utils import regression_to_class
from Plot_cells_to_RP import draw_regression_plane_divisions, setup_plot
from Plot_cells_to_RP import colors as colorsforplot

import glob
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm
from PIL import Image

# Base directory
base_dir = r"d:\dev\DVP2\2022_v1_zeroPadded_split_with_test"
base_dir = "/storage01/grexai/datasets/Regplane_data/2022_v1_zeroPadded_split_with_test/"
split = "test"
# Choose dataset split (train, val, test)
split = "train"  # Change to "train" or "test" if needed
split = "val"


# Paths
image_dir = os.path.join(base_dir, split, "images")
label2_dir = os.path.join(base_dir, split, "labels2")

# Load all JSON label files
json_files = glob.glob(os.path.join(label2_dir, "*.json"))

# Create a blank canvas (white)
canvas_size = (10000, 10000, 3)  # 10000x10000 with 3 color channels (RGB)
canvas = np.ones(canvas_size, dtype=np.uint8) * 255  # White background

# Visualization parameters
thumb_size = 299  # Thumbnail size for each image
scatter_color = (255, 0, 0)  # Red dots for missing images

# Define colors for different classes (dummy example, replace with actual values)

fig, ax = plt.subplots(figsize=(8, 8))

for json_file in tqdm(json_files):
    with open(json_file, "r") as f:
        data = json.load(f)

    x, y = data["x"], data["y"]
    preds_class = data.get("class", 1)  # Default class is 1 if missing
    preds_class = regression_to_class([(x, y)])[0]-2
    img_name = os.path.basename(json_file).replace(".json", ".jpg")
    img_path = os.path.join(image_dir, img_name)

    if not os.path.exists(img_path):
        img_name = img_name.replace(".jpg", ".png")
        img_path = os.path.join(image_dir, img_name)

    if os.path.exists(img_path):
        # Load image
        img = Image.open(img_path)
        img.thumbnail((thumb_size, thumb_size))

        # Convert image to NumPy array
        cropped_image = np.array(img)

        # Get class color

        # class_color = colorsforplot[(preds_class - 1) % len(colorsforplot)]
        class_color = colorsforplot[40-preds_class-10]
        
        
        # Convert to BGR for OpenCV
        cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

        # Add frame (rectangle border)
        cv2.rectangle(
            cropped_image_bgr,
            (0, 0),
            (cropped_image_bgr.shape[1] - 1, cropped_image_bgr.shape[0] - 1),
            (int(class_color[2]), int(class_color[1]), int(class_color[0])),
            50  # Border thickness
        )

        # Compute distance from center
        distance_to_center = np.linalg.norm(np.array([x, y]) - np.array([5000, 5000]))

        if distance_to_center < 3500:
            # Convert to grayscale
            cropped_image_gray = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2GRAY)

            # Convert back to BGR for consistency
            cropped_image_bgr = cv2.cvtColor(cropped_image_gray, cv2.COLOR_GRAY2BGR)

        # Convert to RGB for displaying in Matplotlib
        cropped_image_rgb = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2RGB)

        # Show image using Matplotlib
        imagebox = OffsetImage(cropped_image_rgb, zoom=0.1)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=1)
        ax.add_artist(ab)
    else:
        # Draw a red dot if image is missing
        ax.scatter(x, y, c='red', s=50, zorder=2)

# Convert canvas to PIL image and display

draw_regression_plane_divisions(ax)

ax.set_title(f"{split.upper()} Data Visualization", color='white')
setup_plot(fig, ax)
# plt.show()
plt.savefig(f"{split.upper()}_data_Visualization.png")
plt.savefig(f"{split.upper()}_data_Visualization.pdf")
