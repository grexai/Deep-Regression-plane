import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import glob
from tqdm import tqdm
from regressionplane_utils import regression_to_class
# Custom dataset class for images with JSON annotations from labels2
class ImageCoordinatesDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform

        # Collect all image files
        self.image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(
            os.path.join(image_folder, "*.png"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        json_path = os.path.join(self.annotation_folder, img_name.replace(".jpg", ".json").replace(".png", ".json"))

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Load the corresponding JSON file
        with open(json_path, "r") as f:
            data = json.load(f)
        # origo centered data
        x, y = data["x"], data["y"]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Normalize (x, y) to [0, 1] for better learning
        target = torch.tensor([x, y], dtype=torch.float32)
        return image, target, img_name  # Include img_name for saving predictions


transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
    transforms.RandomVerticalFlip(p=0.5),    # 50% chance to flip vertically
    transforms.RandomAffine(
        degrees=90,              # Random rotation between -30 and 30 degrees
        translate=(0.1, 0.1),    # Random translation up to 10% of image size
        scale=(0.9, 1.1),        # Random scaling between 80% and 120%
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


transform_val = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths
base_dir = r"d:\dev\DVP2\2022_v1_zeroPadded_split_with_test"

base_dir = "/storage01/grexai/datasets/Regplane_data/2022_v1_zeroPadded_split_with_test"


train_image_folder = os.path.join(base_dir, "trainBalAug_v2_2", "images")
train_annotation_folder = os.path.join(base_dir, "trainBalAug_v2_2", "labels2")
# train_image_folder = os.path.join(base_dir, "train", "images")
# train_annotation_folder = os.path.join(base_dir,"train", "labels2")
val_image_folder = os.path.join(base_dir, "val", "images")
val_annotation_folder = os.path.join(base_dir, "val", "labels2")

# Create dataset and dataloaders
train_dataset = ImageCoordinatesDataset(train_image_folder, train_annotation_folder, transform)
val_dataset = ImageCoordinatesDataset(val_image_folder, val_annotation_folder, transform_val)

train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=192, shuffle=False)

# Load pretrained InceptionV3 model
model = models.inception_v3(pretrained=True)
model.aux_logits = False  # Disable auxiliary classifier

# Modify final layer for (x, y) regression
model.fc = nn.Linear(model.fc.in_features, 2)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
num_epochs = 100
best_val_loss = float('inf')
best_model_path = "best_model_inception_L1.pth"
final_model_path = "final_model_inception_L1.pth"
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for images, targets, _ in tqdm(train_loader):  # Ignore img_name during training
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation Step
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    # Print epoch results
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save the best 
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at epoch {epoch + 1}")
    
torch.save(model.state_dict(), final_model_path)
print("Training complete!")


# Inference function and save predictions
def predict_and_save(model, image_folder, output_json="predictions.json"):
    model.eval()
    predictions = {}

    image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))
    

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        json_path = img_path.replace("images", "labels2").replace("jpg", "json")

        with open(json_path, "r") as f:
            current_gt = json.load(f)
        
        with torch.no_grad():
            pred = model(image).cpu().numpy()[0]  # Convert back to original scale
        
        
        # Save predictions
        predictions[img_name] = {"x": int(pred[0]),
                                "y": int(pred[1]),
                                "class":regression_to_class([pred],shift=90)[0],
                                "gt_class":regression_to_class([(current_gt["x"],current_gt["y"])],shift=90)[0]}

    # Write to JSON
    with open(output_json, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to {output_json}")


# Load the best saved model before running inference
model.load_state_dict(torch.load(best_model_path))
print(f"Loaded best model from {best_model_path}")
# print(model.state_dict())
# Run predictions on validation set and save results
predict_and_save(model, val_image_folder, "val_inception_pred_predictions.json")
