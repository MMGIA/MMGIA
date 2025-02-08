import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Resolve OpenMP issue

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load Pre-trained Model
model = models.resnet50(pretrained=True)
model.eval()

# Define Loss Function
criterion = nn.CrossEntropyLoss()

# Load Local Image
image_path = "D:\datasets\test.jpg"  # Replace with your local image path
image = Image.open(image_path).convert('RGB')

# Preprocess Image
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

true_data = transform(image).unsqueeze(0)  # Add batch dimension
target = torch.tensor([0])  # Replace with a dummy label (can be adjusted)

# Simulated Gradient Sharing
output = model(true_data)
loss = criterion(output, target)
loss.backward()
gradients = [param.grad.clone() for param in model.parameters() if param.grad is not None]

# Gradient Inversion Attack (DLG Base)
def gradient_inversion(model, gradients, shape, iterations=1000, lr=0.1):
    reconstructed_data = torch.randn(shape, requires_grad=True)
    optimizer = optim.Adam([reconstructed_data], lr=lr)

    for i in range(iterations):
        optimizer.zero_grad()

        # Forward pass using reconstructed data
        output = model(reconstructed_data)
        loss = criterion(output, target)

        # Calculate gradients
        fake_gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        fake_gradients = [g for g in fake_gradients if g is not None]

        # Gradient matching loss
        gradient_loss = sum(torch.norm(fake_g - real_g) for fake_g, real_g in zip(fake_gradients, gradients))

        gradient_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {gradient_loss.item()}")
            # Visualize intermediate results
            reconstructed_image = reconstructed_data.detach().squeeze().permute(1, 2, 0).numpy()
            reconstructed_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())  # Normalize to [0, 1]
            plt.imshow(reconstructed_image)
            plt.title(f"Reconstructed Image at Iteration {i}")
            plt.show()

    return reconstructed_data

# Perform Attack
shape = true_data.shape
reconstructed_data = gradient_inversion(model, gradients, shape)

# Visualize Final Result
reconstructed_image = reconstructed_data.detach().squeeze().permute(1, 2, 0).numpy()
reconstructed_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())  # Normalize to [0, 1]
plt.imshow(reconstructed_image)
plt.title("Final Reconstructed Image")
plt.show()
