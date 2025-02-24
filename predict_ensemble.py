import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
import glob
import torch.nn as nn
from regressionplane_utils import regression_to_class
import matplotlib.pyplot
import numpy as np


def load_regression_plane_ensemble_models(incetion_path,  resnet_path,device):
    # Load InceptionV3 model with 1000 classes (original state_dict)
    inception_model = models.inception_v3(pretrained=True)

    inception_model.aux_logits = False
    inception_model.fc = nn.Linear(inception_model.fc.in_features, 2)  # Modify output layer
    inception_model.load_state_dict(torch.load(incetion_path))
    inception_model.to(device)
    inception_model.eval()
    # print("InceptionV3 Last Layer Weights:", inception.state_dict())
    # Load ResNet50 model for 41 classes
    resnet_model = models.resnet50(pretrained=True)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 41)
    resnet_model.load_state_dict(torch.load(resnet_path))
    # Modify output layer
    resnet_model.to(device)
    resnet_model.eval()    
    
    
    
    return inception_model, resnet_model

def predict_regression_plane_ensemble_models(inceptionmodel,
                                            resnetmodel, 
                                            image_inception,
                                            image_resnet,
                                            allowed_class_diff=3):
    # Inference
    with torch.no_grad():
        pred_xy = inceptionmodel(image_inception).cpu().numpy()
        
        pred_class = resnetmodel(image_resnet)
        most_probable_classes_resnet = torch.argmax(pred_class, dim=1).cpu().numpy()        # Convert predicted coordinates to class
    
    pred_class_from_coords = regression_to_class(pred_xy,shift=90)
   
    diff = np.abs(most_probable_classes_resnet - pred_class_from_coords)

    # Adjust for the cases where the difference exceeds 20 and wrap around to 41
    diff[diff > 20] = np.abs(diff[diff > 20] - 41)

    # Determine where the difference is within the allowed range
    agreementIdx = diff <= allowed_class_diff

    # Apply the filtering: set pred_xy to None where the class difference is too large
    filtered_pred_xy = np.array([pred_xy[i] if agreementIdx[i] else [np.nan,np.nan] for i in range(len(pred_xy))])
    # Create a mask for coordinates that are not None
    mask = ~np.isnan(np.array(filtered_pred_xy)[:, 0])

    # Get indices where the mask is True
    filtered_pred_indices = np.where(mask)[0]
    return filtered_pred_xy,filtered_pred_indices, np.array(pred_class_from_coords), np.array(most_probable_classes_resnet)
    

if __name__== "__main__":
    # Paths
    image_folder = "/storage01/grexai/datasets/Regplane_data/2022_v1_zeroPadded_split_with_test/val/images/"
    label_folder = "/storage01/grexai/datasets/Regplane_data/2022_v1_zeroPadded_split_with_test/val/labels2/"
    # models
    regression_model_path = "./best_model_inception_L1.pth"
    classification_model_path = "./best_model_resnet50.pth"
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Define transforms
    transform_inception = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_resnet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    inception, resnet = load_regression_plane_ensemble_models(regression_model_path, classification_model_path,device=device)


    # Collect test images
    test_images = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))

    # Run inference
    print("Running predictions...")
    fig, ax = plt.subplots()
    error_list = []
    for i,img_path in enumerate(test_images[:]):
        img_name = os.path.basename(img_path)
        json_path = os.path.join(label_folder, img_name.replace(".jpg", ".json").replace(".png", ".json"))

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load ground truth label
        with open(json_path, "r") as f:
            label_data = json.load(f)

        x_gt, y_gt = label_data["x"], label_data["y"]
        gt_class = regression_to_class([(x_gt, y_gt)],shift=90)[0]

        # Prepare images for both models
        img_inception = transform_inception(image).unsqueeze(0).to(device)

        img_resnet = transform_resnet(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            pred_xy = inception(img_inception).cpu().numpy()[0]
            pred_class = resnet(img_resnet).argmax(dim=1).item()
            # Convert predicted coordinates to class
        pred_class_from_coords = regression_to_class([pred_xy],shift=90)[0]
            # Print results
        print(f"Image: {img_name}")
        print(f"  Ground Truth - Class: {gt_class}, Coordinates: ({x_gt}, {y_gt})")
        print(f"  InceptionV3  - Predicted Coordinates: ({int(pred_xy[0])}, {int(pred_xy[1])}),cls {pred_class_from_coords}")
        print(f"  ResNet-50    - Predicted Class: {pred_class}")
        print("-" * 80)
                
        # Offset for the annotation text
        offset_x = 20  # Horizontal offset
        offset_y = 20  # Vertical offset

        # Draw the line between GT and pred
        ax.plot([x_gt, pred_xy[0]], [y_gt, pred_xy[1]], color='green', linestyle='-', linewidth=1)

        # Scatter for ground truth point and annotation
        ax.scatter(x_gt, y_gt, color='red')
        ax.annotate(f"{i},{gt_class}", (x_gt + offset_x, y_gt + offset_y))

        # Scatter for predicted point and annotation
        ax.scatter(pred_xy[0], pred_xy[1], color='blue')
        #ax.annotate(f"{i}, {pred_class_from_coords}", (pred_xy[0] + offset_x, pred_xy[1] + offset_y))
        gt = np.array([x_gt, y_gt])
        pred = np.array([pred_xy[0], pred_xy[1]])

        # Calculate the Euclidean distance between the ground truth and predicted coordinates
        distance = np.linalg.norm(gt - pred)
        error_list.append(distance)
        # ax.annotate(f"{i}, {pred_class_from_coords}", (pred_xy[0], pred_xy[1]))
        #plt.scatter(x_gt,y_gt,label)
    error_list= np.array(error_list)
    print(f"mean squared error is: {np.sum(error_list)/error_list.shape[0]}")
    plt.xlim(0,10000)
    plt.ylim(0,10000)
    ax.scatter(5000, 5000)
    plt.savefig("compare gt.jpg")

