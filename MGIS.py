import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from transformers import BertTokenizer, BertModel
import numpy as np

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Image and Text Models
image_model = models.resnet50(pretrained=True).to(device)
text_model = BertModel.from_pretrained("bert-base-uncased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_text(text):
    return tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)


# MGIS Attack Class
class MGISAttack(nn.Module):
    def __init__(self, image_model, text_model):
        super(MGISAttack, self).__init__()
        self.image_model = image_model
        self.text_model = text_model

    def attack_text(self, text_gradients, iterations=200, lr=0.01):
        dummy_text = torch.randn(1, 512, requires_grad=True, device=device)
        optimizer_txt = optim.Adam([dummy_text], lr=lr)

        for _ in range(iterations):
            optimizer_txt.zero_grad()
            dummy_output = self.text_model(inputs_embeds=dummy_text).pooler_output
            dummy_loss = dummy_output.sum()
            dummy_grads = torch.autograd.grad(dummy_loss, self.text_model.parameters(), create_graph=True)
            loss = sum(torch.nn.functional.mse_loss(dg, tg) for dg, tg in zip(dummy_grads, text_gradients))
            loss.backward()
            optimizer_txt.step()

        return dummy_text.detach()

    def attack_image(self, image_gradients, text_labels, iterations=200, lr=0.01):
        dummy_image = torch.randn(1, 3, 224, 224, requires_grad=True, device=device)
        optimizer_img = optim.Adam([dummy_image], lr=lr)

        for _ in range(iterations):
            optimizer_img.zero_grad()
            dummy_output = self.image_model(dummy_image)
            dummy_loss = (dummy_output * text_labels).sum()
            dummy_grads = torch.autograd.grad(dummy_loss, self.image_model.parameters(), create_graph=True)
            loss = sum(torch.nn.functional.mse_loss(dg, ig) for dg, ig in zip(dummy_grads, image_gradients))
            loss.backward()
            optimizer_img.step()

        return dummy_image.detach()


# Running MGIS Attack
image_gradients = torch.randn_like(torch.rand(1, 3, 224, 224)).to(device)
text_gradients = torch.randn_like(torch.rand(1, 512)).to(device)

mgis_attack = MGISAttack(image_model, text_model).to(device)

# Step 1: Attack Text Modality
recovered_text_labels = mgis_attack.attack_text(text_gradients)

# Step 2: Use Recovered Text Labels to Guide Image Attack
recovered_image = mgis_attack.attack_image(image_gradients, recovered_text_labels)

print("MGIS attack executed successfully: text labels used to enhance image reconstruction.")
