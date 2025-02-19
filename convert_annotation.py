import os
import json
import numpy as np
import tifffile as tiff
from glob import glob

# Base directory containing train, val, and test sets
base_dir = r"d:\dev\DVP2\2022_v1_zeroPadded_split_with_test"
base_dir = "/storage01/grexai/datasets/Regplane_data/2022_v1_zeroPadded_split_with_test"

# Function to extract (x, y) from label TIFF
def extract_xy_from_tiff(label_path):
    label_img = tiff.imread(label_path)  # Load the TIFF image

    if label_img.shape[1] < 2:
        print(f"Warning: Less than 2 coordinate points found in {label_path}")
        return None  # Skip if we don't have at least 2 points

    x, y = label_img[0][0], label_img[0][1]  # Take the first two pixels as (x, y)
    return {"x": int(x), "y": int(y)}


def rotate_90_ccw_around_5000(coord):
    """Rotate (x, y) 90 degrees counterclockwise around (5000, 5000)."""
    x, y = coord["x"], coord["y"]

    # Translate so that (5000, 5000) is the origin
    x_shifted, y_shifted = x - 5000, y - 5000

    # Apply 90-degree CCW rotation: (x', y') = (-y, x)
    x_rotated = -y_shifted
    y_rotated = x_shifted

    # Translate back to original coordinate space
    x_new = x_rotated + 5000
    y_new = y_rotated + 5000

    return {"x": int(x_new), "y": int(y_new)}



# Process each dataset split
for split in ["trainBalAug_v2_2", "val", "test"]:
    image_dir = os.path.join(base_dir, split, "images")
    label_dir = os.path.join(base_dir, split, "labels")
    label2_dir = os.path.join(base_dir, split, "labels2")  # New folder for JSONs

    # Create labels2 folder if it doesn't exist
    os.makedirs(label2_dir, exist_ok=True)

    # Get all image file names (assuming .jpg or .png images)
    image_files = glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png"))

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".tif").replace(".png", ".tiff"))

        if not os.path.exists(label_path):
            print(f"Label file missing for {img_name}, skipping...")
            continue

        # Extract (x, y) coordinates
        xy_data = extract_xy_from_tiff(label_path)
        xy_data = rotate_90_ccw_around_5000(xy_data)

        if xy_data is None:
            continue

        # Save to JSON in labels2
        json_path = os.path.join(label2_dir, img_name.replace(".jpg", ".json").replace(".png", ".json"))
        with open(json_path, "w") as f:
            json.dump(xy_data, f, indent=4)

        print(f"Saved {json_path}")

print("Processing complete!")
