import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
import cv2  # Import OpenCV
import glob
from tqdm import tqdm

"""
This script draws cell images onto the regression plane 
RP defined as 270 +-15 dgrees as interphase
the rest 255 -> 285  angles dived into 40 subphase 

"""

# Define the color palette
colors = [(73, 89, 159), (90, 87, 157,), (107, 86, 155), (121, 84, 153), (136, 81, 151), (147, 78, 148),
          (158, 74, 146), (170, 69, 144), (180, 65, 141), (190, 59, 138), (200, 53, 136), (208, 48, 134),
          (218, 38, 131), (227, 24, 128), (229, 20, 125), (229, 21, 121), (229, 22, 118), (230, 23, 113),
          (230, 24, 109), (230, 24, 105), (231, 27, 101), (230, 28, 96), (231, 33, 93), (230, 33, 87),
          (230, 34, 82), (230, 32, 78), (230, 33, 72), (230, 38, 70), (231, 61, 71), (232, 82, 71),
          (234, 100, 73), (236, 116, 73), (238, 132, 72), (240, 146, 72), (243, 174, 66), (243, 180, 64),
          (244, 185, 61), (246, 198, 54), (249, 211, 45), (249, 221, 34), (251, 232, 23)]
colorsforplot = [(r / 255, g / 255, b / 255) for r, g, b in colors]


def setup_plot(fig, ax):
    """
    :param fig
    :param ax:
    :return: Nones
    """
    # Plot labels
    ax.set_xlabel('Regression plane position (X)')
    ax.set_ylabel('Regression plane position (Y)')


    # Set limits
    ax.set_xlim(-1000, 11000)
    ax.set_ylim(-1000, 11000)
    # Set axis to be tight
    ax.set_aspect('equal')  # Keep aspect ratio square
    plt.tight_layout()  # Adjust layout for a tight fit
    # Set black background
    fig.patch.set_facecolor('black')  # Figure background
    ax.set_facecolor('black')  # Axes background

    # Change axis labels and ticks to white for visibility
    ax.tick_params(colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.xaxis.label.set_color('white')  # X-axis label color
    ax.yaxis.label.set_color('white')  # Y-axis label color
    ax.tick_params(colors='white')  # Tick color
    # Show the plot


def draw_regression_plane_divisions(ax, center=(5000, 5000), inner_radius=3500, outer_radius=5000, text_radius=5200,
                                    line_radius=5400, zorder=1):
    """
    Plots circular divisions on a given axis.

    Parameters:
        ax : matplotlib.axes.Axes
            The axis on which to plot.
        center : tuple
            The center of the circles.
        inner_radius : int
            The inner radius for the division lines.
        outer_radius : int
            The outer radius for the reference circles.
        text_radius : int
            The radius at which text labels are placed.
        line_radius : int
            The radius at which division lines end.
        zorder : int
            Z-order of the plotted elements.
    """
    # Draw reference circles
    for radius in [inner_radius, outer_radius]:
        ax.add_artist(
            plt.Circle(center, radius, color='#D3D3D3', fill=False, linestyle='-', linewidth=2, zorder=zorder))

    # Define angles
    bottom_section_angles = [255, 285]
    remaining_angles = np.linspace(-75, 255, 41) % 360  # 40 divisions

    # Combine and process angles
    angles = np.sort(np.concatenate((bottom_section_angles, remaining_angles)))
    angles = np.unique(angles)[::-1]
    angles = np.roll(angles, -10)

    # Plot division lines and labels
    for idx, angle in enumerate(angles):
        angle_rad = np.deg2rad(angle)
        shift_angle = np.deg2rad(8.25 / 2)

        # Calculate start and end points for lines
        start_point = np.array(center) + inner_radius * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        end_point = np.array(center) + line_radius * np.array([np.cos(angle_rad), np.sin(angle_rad)])

        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='#D3D3D3', linewidth=1,
                zorder=zorder)

        # Calculate text label position
        label_position = np.array(center) + text_radius * np.array(
            [np.cos(angle_rad + shift_angle), np.sin(angle_rad + shift_angle)])
        if idx == 0:
            label_position = np.array([5000, -300])

        ax.text(label_position[0], label_position[1], str(idx + 1), color='white', fontsize=10, ha='center',
                va='center')


if __name__ == "__main__":

    # Define the paths
    csv_dir = r'D:\datasets\dvp2_sample_DRP\Proteomics-acc\221123-HK-DVP2-frame-60X-56__2022-11-23T11_30_34-Measurement1\predictions4BIAS_bulk'
    image_dir = r'D:\datasets\dvp2_sample_DRP\Proteomics-acc\221123-HK-DVP2-frame-60X-56__2022-11-23T11_30_34-Measurement1\anal3'

    path = "e:/DVP2/acc"
    exp_name = "220908-HK-DVP2-frame-60x-40-2__2022-09-08T19_01_02-Measurement"
    csv_dir = os.path.join(path)
    # Use glob to find all matching files
    file_pattern = os.path.join(path, f"{exp_name}*.csv")  # Matches files like "exp_name_1.csv"
    matching_files = glob.glob(file_pattern)

    # Read and concatenate all matching CSV files
    df_list = [pd.read_csv(file) for file in matching_files]
    df = pd.concat(df_list, ignore_index=True)

    image_dir = os.path.join(path, f"{exp_name}1", "anal3")
    output_dir = os.path.join(path, exp_name, f"cropped_{exp_name}")
    # Read CSV
    # "220908-HK-DVP2-frame-60x-40-2__2022-09-08T19_01_02-Measurement_1.csv"
    # csv_path = os.path.join(csv_dir, '221123-HK-DVP2-frame-60X-56__2022-11-23T11_30_34-Measurement1.csv')
    # csv_path = os.path.join(csv_dir, f"{exp_name}_1.csv")
    # df = pd.read_csv(csv_path)


    # df = df[df["PredsRegIdx"] != 1]  # Remove certain indexes

    # Select all rows where PredsRegIdx == 1
    class_1_rows = df[df["PredsRegIdx"] == 1]

    # Sample 1% of class 1 rows (ensure at least one row is kept if applicable)
    sampled_class_1 = class_1_rows.sample(frac=0.01, random_state=42)  # Adjust `random_state` for reproducibility

    # Keep all other rows
    df_other = df[df["PredsRegIdx"] != 1]

    # Combine back
    df = pd.concat([df_other, sampled_class_1], ignore_index=True)

    zorder_of_drawings = 100
    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 8))

    draw_regression_plane_divisions(ax)

    # Iterate through each row in CSV
    idx = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        idx = idx + 1
        if idx == 1000:
            break
        preds_x = row['PredsRegPos_1']  # X center
        preds_y = row['PredsRegPos_2']  # Y center
        preds_class = row['PredsRegIdx']

        # Bounding box extraction
        xmin, ymin, xmax, ymax = row['Boundingbox_1'], row['Boundingbox_2'], row['Boundingbox_3'], row['Boundingbox_4']
        filename = row['Filename']
        image_path = os.path.join(image_dir, f'{filename}.tif')

        if not os.path.exists(image_path):
            continue

        # Read the image with skimage (use OpenCV here if you prefer)
        image = io.imread(image_path)

        # Validate bounding box within image limits
        if xmin < 0 or ymin < 0 or xmax > image.shape[1] or ymax > image.shape[0]:
            print(f"Skipping invalid bounding box for {filename}: {(xmin, ymin, xmax, ymax)}")
            continue

        # Crop the image
        cropped_image = image[ymin:ymax, xmin:xmax]
        if cropped_image.size == 0:
            print(f"Skipping empty crop for {filename}: {(xmin, ymin, xmax, ymax)}")
            continue

        # Get class color
        class_color = colorsforplot[preds_class - 1 % len(colorsforplot)]

        # Add a frame around the cropped image using OpenCV (add a rectangle border)
        # Convert the image to BGR for OpenCV to work correctly with colors
        cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

        # Draw the frame (rectangle) around the cropped image
        # cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(cropped_image_bgr, (0, 0), (cropped_image_bgr.shape[1] - 1, cropped_image_bgr.shape[0] - 1),
                      (int(class_color[0] * 255), int(class_color[1] * 255), int(class_color[2] * 255)), 50)

        # Convert to RGB for displaying in Matplotlib
        cropped_image_rgb = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2RGB)

        distance_to_center = np.linalg.norm(np.array([preds_x, preds_y]) - np.array([5000, 5000]))

        if distance_to_center < 3500:
            # Convert to grayscale
            cropped_image_gray = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2GRAY)

            # Convert back to RGB (3 channels)
            cropped_image_bgr = cv2.cvtColor(cropped_image_gray, cv2.COLOR_GRAY2BGR)

        # Show cropped image at (X, Y) center with a frame
        imagebox = OffsetImage(cropped_image_bgr, zoom=0.1)
        ab = AnnotationBbox(imagebox, (preds_x, preds_y), frameon=False, zorder=1)
        ax.add_artist(ab)

    # Plot labels
    ax.set_xlabel('Regression plane position (X)')
    ax.set_ylabel('Regression plane position (Y)')
    ax.set_title('Predicted cells on Regression plane', color='white')


    # Set limits
    ax.set_xlim(-1000, 11000)
    ax.set_ylim(-1000, 11000)
    # Set axis to be tight
    ax.set_aspect('equal')  # Keep aspect ratio square
    plt.tight_layout()  # Adjust layout for a tight fit
    # Set black background
    fig.patch.set_facecolor('black')  # Figure background
    ax.set_facecolor('black')  # Axes background
    # Change axis labels and ticks to white for visibility
    ax.tick_params(colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.xaxis.label.set_color('white')  # X-axis label color
    ax.yaxis.label.set_color('white')  # Y-axis label color
    ax.tick_params(colors='white')  # Tick color
    # Show the plot
    plt.show()
