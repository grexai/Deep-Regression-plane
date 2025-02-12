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
# Paths
image_folder = "d:/dev/DVP2/2022_v1_zeroPadded_split_with_test/val/images/"
label_folder = "d:/dev/DVP2/2022_v1_zeroPadded_split_with_test/val/labels2/"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform_inception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load InceptionV3 model with 1000 classes (original state_dict)
inception = models.inception_v3(pretrained=True)

inception.aux_logits = False
inception.fc = nn.Linear(inception.fc.in_features, 2)  # Modify output layer
inception.load_state_dict(torch.load('./best_model_inception.pth'))
inception.to(device)
inception.eval()
# print("InceptionV3 Last Layer Weights:", inception.state_dict())
# Load ResNet50 model for 41 classes
resnet = models.resnet50(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 41)
resnet.load_state_dict(torch.load('./best_model_resnet50.pth'))
  # Modify output layer
resnet.to(device)
resnet.eval()
# Collect test images
test_images = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))

# Run inference
print("Running predictions...")
fig, ax = plt.subplots()

for img_path in test_images[:]:
    img_name = os.path.basename(img_path)
    json_path = os.path.join(label_folder, img_name.replace(".jpg", ".json").replace(".png", ".json"))

    # Load image
    image = Image.open(img_path).convert("RGB")

    # Load ground truth label
    with open(json_path, "r") as f:
        label_data = json.load(f)

    x_gt, y_gt = label_data["x"], label_data["y"]
    gt_class = regression_to_class((x_gt, y_gt))

    # Prepare images for both models
    img_inception = transform_inception(image).unsqueeze(0).to(device)

    img_resnet = transform_resnet(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred_xy = inception(img_inception).cpu().numpy()[0]*10000.0
        pred_class = resnet(img_resnet).argmax(dim=1).item()
        # Convert predicted coordinates to class
    pred_class_from_coords = regression_to_class((pred_xy[0], pred_xy[1]))
        # Print results
    print(f"Image: {img_name}")
    print(f"  Ground Truth - Class: {gt_class}, Coordinates: ({x_gt}, {y_gt})")
    print(f"  InceptionV3  - Predicted Coordinates: ({int(pred_xy[0])}, {int(pred_xy[1])}),cls {pred_class_from_coords}")
    print(f"  ResNet-50    - Predicted Class: {pred_class}")
    print("-" * 80)
    ax.scatter(x_gt, y_gt)
    ax.annotate(gt_class,(x_gt, y_gt))
    #plt.scatter(x_gt,y_gt,label)
plt.xlim(0,10000)
plt.ylim(0,10000)
ax.scatter(5000, 5000)
plt.show()