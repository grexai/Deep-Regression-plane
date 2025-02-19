import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from regressionplane_utils import regression_to_class
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Define paths
base_dir = r"d:\dev\DVP2\2022_v1_zeroPadded_split_with_test"
base_dir = "/storage01/grexai/datasets/Regplane_data/2022_v1_zeroPadded_split_with_test"
train_image_folder = os.path.join(base_dir, "trainBalAug_v2_2", "images")
train_annotation_folder = os.path.join(base_dir, "trainBalAug_v2_2", "labels2")
# train_image_folder = os.path.join(base_dir, "train", "images")
# train_annotation_folder = os.path.join(base_dir,"train", "labels2")
val_image_folder = os.path.join(base_dir, "val", "images")
val_annotation_folder = os.path.join(base_dir, "val", "labels2")


# Define dataset class
class MitoticDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_folder))
        self.label_filenames = sorted(os.listdir(annotation_folder))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        label_path = os.path.join(self.annotation_folder, self.label_filenames[idx])

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Load label
        with open(label_path, 'r') as f:
            label_data = json.load(f)
            x, y = label_data["x"], label_data["y"]
            class_label = regression_to_class([(x, y)],shift=90)[0]
            class_label = torch.tensor(class_label, dtype=torch.long)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, class_label


# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
    transforms.RandomVerticalFlip(p=0.5),    # 50% chance to flip vertically
    transforms.RandomAffine(
        degrees=90,              # Random rotation between -30 and 30 degrees
        translate=(0.1, 0.1),    # Random translation up to 10% of image size
        scale=(0.9, 1.1),        # Random scaling between 80% and 120%
    ),
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloaders
train_dataset = MitoticDataset(train_image_folder, train_annotation_folder, transform=transform)
val_dataset = MitoticDataset(val_image_folder, val_annotation_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Load ResNet-50 model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 41)  # 41 classes

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop with validation
num_epochs = 10
best_val_loss = float('inf')
best_model_path = "best_model_resnet50.pth"
final_model_path = 'final_model_resnet50.pth'
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation step
    model.eval()
    val_loss = 0.0
    model.eval()
val_loss = 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        # Get predictions
        preds = torch.argmax(outputs, dim=1)

        # Store predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Compute loss
        avg_val_loss = val_loss / len(val_loader)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch + 1} with Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
torch.save(model.state_dict(), final_model_path)
print(f"final model saved {final_model_path}")
print("Training complete!")
