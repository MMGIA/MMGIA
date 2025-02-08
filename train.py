import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from transformers import BertTokenizer, BertModel
import os

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # CIFAR100 dataset (commented out for now)
    # cifar100 = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)

    # Flickr30K dataset (commented out for now)
    # flickr30k = datasets.FakeData(size=1000, image_size=(3, 224, 224), num_classes=10, transform=transform)  # Placeholder for Flickr30K

    # OpenI dataset (running separately)
    openi = datasets.FakeData(size=1000, image_size=(3, 224, 224), num_classes=10,
                              transform=transform)  # Placeholder for OpenI

    return openi  # Only OpenI is being used for this run


openi = load_datasets()

# Federated Learning Setup
num_clients = 10


def federated_sgd(image_model, text_model, data, num_rounds=5, lr=0.01):
    global_image_model = models.resnet50(pretrained=True).to(device)
    global_text_model = BertModel.from_pretrained("bert-large-uncased").to(device)

    global_image_model.load_state_dict(image_model.state_dict())
    global_text_model.load_state_dict(text_model.state_dict())

    for round in range(num_rounds):
        local_weights_img, local_weights_txt = [], []
        for client in range(num_clients):
            optimizer_img = optim.SGD(image_model.parameters(), lr=lr)
            optimizer_txt = optim.SGD(text_model.parameters(), lr=lr)

            optimizer_img.zero_grad()
            optimizer_txt.zero_grad()

            output_img = image_model(data[client][0])
            output_txt = text_model(**preprocess_text("A sample text for training"))["pooler_output"]

            loss_img = output_img.mean()
            loss_txt = output_txt.mean()

            loss_img.backward()
            loss_txt.backward()

            optimizer_img.step()
            optimizer_txt.step()

            local_weights_img.append(image_model.state_dict())
            local_weights_txt.append(text_model.state_dict())

        # Federated Averaging (FedSGD)
        new_state_dict_img = {key: torch.mean(torch.stack([w[key] for w in local_weights_img]), dim=0) for key in
                              local_weights_img[0]}
        new_state_dict_txt = {key: torch.mean(torch.stack([w[key] for w in local_weights_txt]), dim=0) for key in
                              local_weights_txt[0]}

        global_image_model.load_state_dict(new_state_dict_img)
        global_text_model.load_state_dict(new_state_dict_txt)

    return global_image_model, global_text_model


# Execute Federated Training on OpenI
global_image_model, global_text_model = federated_sgd(image_model, text_model, [openi])

# Save the trained global models
os.makedirs("./output", exist_ok=True)
torch.save(global_image_model.state_dict(), "./output/global_image_model.pth")
torch.save(global_text_model.state_dict(), "./output/global_text_model.pth")
print("Federated training on OpenI completed and models saved.")
