import torch
import torch.nn as nn
import torchvision.models as models
import scipy.io

# Load InceptionV3
model = models.inception_v3(pretrained=False, aux_logits=False)

# Load MATLAB weights
mat_data = scipy.io.loadmat('D:/dev/DVP2/drp/inceptionv3_weights.mat')
print(mat_data.keys())  # Print available keys to verify

# Helper function to load weights
def assign_weights(torch_layer, matlab_weights_key, permute=False):
    if matlab_weights_key in mat_data:
        weight_tensor = torch.tensor(mat_data[matlab_weights_key], dtype=torch.float32)
        if permute:
            weight_tensor = weight_tensor.permute(3, 2, 0, 1)  # MATLAB stores as (H, W, C, N)
        torch_layer.weight.data = weight_tensor

for name, param in model.named_parameters():
    print(name, param.size())

# Assign Convolutional Weights
assign_weights(model.Conv2d_1a_3x3.conv, 'W_conv2d_1', permute=True)
assign_weights(model.Conv2d_2a_3x3.conv, 'W_conv2d_2', permute=True)

# Assign Fully Connected Layer (Check correct naming)
if 'W_FcOut_1' in mat_data and 'B_FcOut_1' in mat_data:
    model.fc.weight.data = torch.tensor(mat_data['W_FcOut_1'], dtype=torch.float32)
    model.fc.bias.data = torch.tensor(mat_data['B_FcOut_1'], dtype=torch.float32)
else:
    print("⚠️ Warning: Fully connected layer weights not found in .mat file!")

# Save to Torch Model
torch.save(model.state_dict(), "inceptionv3_converted.pth")
print("✅ Weights converted successfully!")
