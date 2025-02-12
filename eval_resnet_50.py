import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from regressionplane_utils import regression_to_class
from PIL import Image
import os
import json

# Define paths
base_dir = r"d:\dev\DVP2\2022_v1_zeroPadded_split_with_test"
val_image_folder = os.path.join(base_dir, "val", "images")
val_annotation_folder = os.path.join(base_dir, "val", "labels2")
test_image_folder = os.path.join(base_dir, "test", "images")
test_annotation_folder = os.path.join(base_dir, "test", "labels2")


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

        image = Image.open(img_path).convert('RGB')
        with open(label_path, 'r') as f:
            label_data = json.load(f)
            x, y = label_data["x"], label_data["y"]
            class_label = regression_to_class((x, y))
            class_label = torch.tensor(class_label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, class_label


# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
datasets = {
    "val": MitoticDataset(val_image_folder, val_annotation_folder, transform=transform),
    "test": MitoticDataset(test_image_folder, test_annotation_folder, transform=transform)
}

dataloaders = {key: DataLoader(datasets[key], batch_size=32, shuffle=False) for key in datasets}

# Load trained model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 41)
model.load_state_dict(torch.load("resnet50_trained.pth"))
model.eval()

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Evaluation function
def evaluate_model(model, dataloader, dataset_name):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"{dataset_name} Accuracy: {accuracy:.2f}%")


# Run evaluation
for dataset_name, dataloader in dataloaders.items():
    evaluate_model(model, dataloader, dataset_name)
