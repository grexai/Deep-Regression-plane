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
        x, y = data["x"], data["y"]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Normalize (x, y) to [0, 1] for better learning
        target = torch.tensor([x / 10000.0, y / 10000.0], dtype=torch.float32)
        return image, target, img_name  # Include img_name for saving predictions


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Paths
base_dir = r"d:\dev\DVP2\2022_v1_zeroPadded_split_with_test"
train_image_folder = os.path.join(base_dir, "train", "images")
train_annotation_folder = os.path.join(base_dir, "train", "labels2")
val_image_folder = os.path.join(base_dir, "val", "images")
val_annotation_folder = os.path.join(base_dir, "val", "labels2")

# Create dataset and dataloaders
train_dataset = ImageCoordinatesDataset(train_image_folder, train_annotation_folder, transform)
val_dataset = ImageCoordinatesDataset(val_image_folder, val_annotation_folder, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

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
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop with validation
num_epochs = 0
best_val_loss = float('inf')
best_model_path = "best_model_inception.pth"

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for images, targets, _ in train_loader:  # Ignore img_name during training
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

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at epoch {epoch + 1}")

print("Training complete!")


# Inference function and save predictions
def predict_and_save(model, image_folder, annotation_folder, output_json="predictions.json"):
    model.eval()
    predictions = {}

    image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image).cpu().numpy()[0] * 10000  # Convert back to original scale

        # Save predictions
        predictions[img_name] = {"x": int(pred[0]), "y": int(pred[1])}

    # Write to JSON
    with open(output_json, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to {output_json}")


# Load the best saved model before running inference
model.load_state_dict(torch.load(best_model_path))
print(f"Loaded best model from {best_model_path}")
# print(model.state_dict())
# Run predictions on validation set and save results
predict_and_save(model, val_image_folder, val_annotation_folder, "val_predictions.json")
