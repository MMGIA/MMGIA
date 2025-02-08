import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from PIL import Image
import clip
import numpy as np
import os

# Load CLIP Model for multimodal alignment
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Image and Text Models
image_model = models.resnet50(pretrained=True).to(device)
text_model = BertModel.from_pretrained("bert-large-uncased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_text(text):
    return tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)


# Load Datasets
def load_datasets():
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    cifar100 = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    openi = datasets.FakeData(size=1000, image_size=(3, 224, 224), num_classes=10,
                              transform=transform)  # Placeholder for OpenI
    flickr30k = datasets.FakeData(size=1000, image_size=(3, 224, 224), num_classes=10,
                                  transform=transform)  # Placeholder for Flickr30K
    return cifar100, openi, flickr30k


cifar100, openi, flickr30k = load_datasets()


# Gradient Inversion Attack (Stage 1: Single Modality Reconstruction)
def gradient_inversion_attack(model, gradients, dummy_input, optimizer, iterations=200):
    model.eval()
    for _ in range(iterations):
        optimizer.zero_grad()
        dummy_output = model(dummy_input)
        dummy_grads = \
        torch.autograd.grad(outputs=dummy_output, inputs=dummy_input, grad_outputs=gradients, create_graph=True)[0]
        loss = ((dummy_grads - gradients) ** 2).sum()
        loss.backward()
        optimizer.step()
    return dummy_input.detach()


# Stage 2: Multimodal Refinement with CLIP and Quality-Weighted Fusion
class MMGIA(nn.Module):
    def __init__(self, image_model, text_model, clip_model):
        super(MMGIA, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.clip_model = clip_model

    def forward(self, image, text):
        image_features = self.image_model(image)
        text_features = self.text_model(**preprocess_text(text))["pooler_output"]

        # CLIP Embeddings
        with torch.no_grad():
            image_clip_emb = self.clip_model.encode_image(image)
            text_clip_emb = self.clip_model.encode_text(clip.tokenize(text).to(device))

        # Compute quality scores
        image_quality = torch.norm(image_clip_emb, p=2)
        text_quality = torch.norm(text_clip_emb, p=2)
        total_quality = image_quality + text_quality

        # Compute weighted fusion embedding
        image_weight = image_quality / total_quality
        text_weight = text_quality / total_quality
        fused_embedding = image_weight * image_clip_emb + text_weight * text_clip_emb

        # Cross-modal alignment loss
        loss_align = torch.norm(image_clip_emb - fused_embedding, p=2) + torch.norm(text_clip_emb - fused_embedding,
                                                                                    p=2)
        return loss_align


# Running Attack
image_gradients = torch.randn_like(torch.rand(1, 3, 224, 224)).to(device)
text_gradients = torch.randn_like(torch.rand(1, 512)).to(device)

dummy_image = torch.randn(1, 3, 224, 224, requires_grad=True, device=device)
dummy_text = torch.randn(1, 512, requires_grad=True, device=device)

optimizer_img = optim.Adam([dummy_image], lr=0.01)
optimizer_txt = optim.Adam([dummy_text], lr=0.01)

# Stage 1: Single Modality Reconstruction
dummy_image = gradient_inversion_attack(image_model, image_gradients, dummy_image, optimizer_img)
dummy_text = gradient_inversion_attack(text_model, text_gradients, dummy_text, optimizer_txt)

# Stage 2: Multimodal Refinement with Quality-Weighted Fusion
mmgia = MMGIA(image_model, text_model, clip_model).to(device)
loss = mmgia(dummy_image, "a sample caption")
loss.backward()

# Save model and results
os.makedirs("./output", exist_ok=True)
torch.save(mmgia.state_dict(), "./output/mmgia_model.pth")
print("Model and results saved successfully.")
