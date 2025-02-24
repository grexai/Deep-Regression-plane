import os
import numpy as np
import pandas as pd
import cv2
import shutil
from scipy.io import loadmat
import torch
import torchvision.transforms as transforms
from predict_ensemble import load_regression_plane_ensemble_models, predict_regression_plane_ensemble_models

import torch
import torchvision.transforms as transforms

torch.cuda.set_device(2)

# Define paths


def create_gird_image_from_crops(crops):
    # Define number of images per row
    images_per_row = 10
    img_h, img_w, img_c = crops.shape[1:]  # Extract image dimensions

    # Calculate number of full rows and remaining images
    num_full_rows = len(crops) // images_per_row
    num_remaining = len(crops) % images_per_row

    # Pad the last row if necessary
    if num_remaining > 0:
        pad_width = (images_per_row - num_remaining, img_h, img_w, img_c)
        padding = np.zeros(pad_width, dtype=np.uint8)  # Black padding
        crops = np.concatenate([crops, padding], axis=0)

    # Reshape into rows
    rows = [np.concatenate(crops[i:i + images_per_row], axis=1) for i in range(0, len(crops), images_per_row)]

    # Stack rows vertically
    final_image = np.concatenate(rows, axis=0)

    # Save the result
    cv2.imwrite("concatenated.png", final_image)


def filter_coordinates(coords, center=np.array([5000, 5000]), radius_threshold=3500):
    coords = np.array(coords)  # Ensure it's a NumPy array
    distances = np.linalg.norm(coords - center, axis=1)  # Compute Euclidean distances
    coords[distances < radius_threshold] = None  # Set values to None if below threshold
    valid_indices = np.where(distances >= radius_threshold)[0]

    return coords, valid_indices


def convert_image_lists_to_batched_tensor(crops, target_size=(299, 299), device=None):
    """
    Convert a NumPy array of images to a PyTorch batched tensor with resizing, normalization, and device transfer.
    
    Args:
        crops (numpy.ndarray): Input images with shape (batch, H, W, C).
        target_size (tuple): Target size for resizing (H, W).
        device (torch.device or str, optional): Target device ('cuda' or 'cpu'). Defaults to 'cuda' if available.
    
    Returns:
        torch.Tensor: Preprocessed images in shape (batch, C, H, W) on the specified device.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert NumPy array to PyTorch tensor
    crops_tensor = torch.tensor(crops/255.0, dtype=torch.float32)  # Convert to float tensor

    # Permute from (batch, H, W, C) -> (batch, C, H, W)
    crops_tensor = crops_tensor.permute(0, 3, 1, 2)  

    # Resize images
    resize = transforms.Resize(target_size)
    crops_tensor = resize(crops_tensor)

    # Normalize using Inception's normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    crops_tensor = normalize(crops_tensor)

    # Move tensor to the specified device
    crops_tensor = crops_tensor.to(device)

    return crops_tensor


regression_model_path = "./best_model_inception_L1.pth"
classification_model_path = "./best_model_resnet50.pth"
# acc_folder = "d:/datasets/dvp2_sample_DRP/Proteomics-acc/221123-HK-DVP2-frame-60X-56__2022-11-23T11_30_34-Measurement1/"
acc_folder = "/NAS/grexa/dvp2_sample_DRP/Proteomics-acc/221123-HK-DVP2-frame-60X-56__2022-11-23T11_30_34-Measurement1/"
out_folder = os.path.join(acc_folder, "predictions4BIAS_bulk")
filter_agreed_images_dir = os.path.join(acc_folder, "filteredAgreement")

# Parameters
regression_model_input_size = (299, 299)
classification_model_input_size = (224, 224)
allowed_class_diff = 3
minimum_radius = 3500
# some BIAS projects exports the images as BGR instead of RGB
# DONT ASK THANKS
switch_rb_channels = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load models
regression_model, classification_model = load_regression_plane_ensemble_models(regression_model_path, classification_model_path, device= device)

# Ensure output directories exist
os.makedirs(out_folder, exist_ok=True)
os.makedirs(os.path.join(filter_agreed_images_dir, "interphase"), exist_ok=True)
for i in range(1, 41):
    os.makedirs(os.path.join(filter_agreed_images_dir, str(i)), exist_ok=True)

# Processing images
an2_folder = os.path.join(acc_folder, "anal2")
an3_folder = os.path.join(acc_folder, "anal3")
extra_folder = os.path.join(acc_folder, "extra")

image_files = [f for f in os.listdir(an3_folder) if not f.startswith(".")]
output_rows = []

for img_idx, file_name in enumerate(image_files[:]):
    print(f"Processing image {img_idx + 1}/{len(image_files)}: {file_name}")
    
    image_path = os.path.join(an3_folder, file_name)
    image = cv2.imread(image_path)
    
    if switch_rb_channels:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sy, sx, _ = image.shape
    fname, _ = os.path.splitext(file_name)

    features = np.loadtxt(os.path.join(an2_folder, f"{fname}.txt"))
    idx_map = np.loadtxt(os.path.join(extra_folder, f"{fname}.txt"))

    num_cells = features.shape[0]
    bigger_input_size = max(regression_model_input_size, classification_model_input_size)
    crops = np.zeros((num_cells, *bigger_input_size, 3), dtype=np.uint8)
    
    # bounding boxes stored in x1,y1,x2,y2, Top left, bottom right
    bounding_boxes = np.zeros((num_cells, 4))
    
    for cell_idx in range(num_cells):
        cx, cy = features[cell_idx, :2]
        ul = np.array([cy, cx]) - np.floor(np.array(bigger_input_size) / 2).astype(int)
        br = np.array([cy, cx]) + np.floor(np.array(bigger_input_size) / 2).astype(int)

        # Compute padding
        pad_top = int(max(0, -ul[0]))
        pad_left = int(max(0, -ul[1]))
        pad_bottom = int(max(0, br[0] - sy))
        pad_right = int(max(0, br[1] - sx))
        ul = np.maximum(ul, 0)
        br = np.minimum(br, [sy, sx])

        crop = image[int(ul[0]):int(br[0]), int(ul[1]):int(br[1])]
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

        crops[cell_idx, :crop.shape[0], :crop.shape[1], :] = crop
        bounding_boxes[cell_idx] = [ul[1], ul[0], br[1], br[0]]

    inception_image_tensor = convert_image_lists_to_batched_tensor(crops)
    restnet_image_tensor = convert_image_lists_to_batched_tensor(crops,(224,224))
    
    preds, agreement_indices, pred_classes,_ = predict_regression_plane_ensemble_models(regression_model, 
                                                                    classification_model,
                                                                    inception_image_tensor,
                                                                    restnet_image_tensor)
    
    
    preds_filtered, valid_indexes = filter_coordinates(preds)                                                                         



    
    preds_reg_idx_filtered = pred_classes[valid_indexes]
    preds_reg_pos_filtered = preds[valid_indexes]
    bounding_boxes_filtered = bounding_boxes[valid_indexes]


# Unpacking BoundingBox into separate columns
bb1, bb2, bb3, bb4 = zip(*[[int(x) for x in b] for b in bounding_boxes_filtered])
prp1, prp2 = zip(*[[int(x) for x in p] for p in preds_reg_pos_filtered])

# Convert data to DataFrame
df = pd.DataFrame({
    "Filename": [fname] * len(valid_indexes),
    "ObjectID": valid_indexes.astype(int),
    "BoundingBox1": bb1,
    "BoundingBox2": bb2,
    "BoundingBox3": bb3,
    "BoundingBox4": bb4,
    "PredsRegIdx": preds_reg_idx_filtered.astype(int),
    "PredsRegPos1": prp1,
    "PredsRegPos2": prp2
    #"AvgLayerFeatures": [str(list(f)) for f in avg_layer_features.T[idx_list_bool]]
})

    output_rows.append(df)

# Save results
output_df = pd.concat(output_rows, ignore_index=True)
output_csv_path = os.path.join(out_folder, f"{os.path.basename(acc_folder.rstrip(os.sep))}.csv")
output_df.to_csv(output_csv_path, index=False)

print("Processing complete. Results saved to:", output_csv_path)

