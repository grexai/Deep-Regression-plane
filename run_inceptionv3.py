import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import glob

# Paths
model_path = "inceptionv3_converted.pth"
image_folder = "d:/dev/DVP2/2022_v1_zeroPadded_split_with_test/test/images/"

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.inception_v3(pretrained=False, aux_logits=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# Define preprocessing (adjust based on model training)
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to InceptionV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and process images
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))  # Adjust extension if needed
results = {}

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    predicted_class = torch.argmax(output, dim=1).item()
    results[os.path.basename(image_path)] = predicted_class

# Print results
for img, pred in results.items():
    print(f"{img}: Predicted Class {pred}")
