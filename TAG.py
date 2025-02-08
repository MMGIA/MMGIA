import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import numpy as np

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a pre-trained text model (BERT)
model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def preprocess_text(text):
    return tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)


def generate_random_text_embedding():
    return torch.randn((1, 512), requires_grad=True, device=device)


# Define the Text Gradient Attack (TAG) function
def tag_attack(model, target_text, iterations=300, lr=0.1):
    # Preprocess target text
    target_input = preprocess_text(target_text)
    target_output = model(**target_input).pooler_output
    target_output.requires_grad = False

    # Compute true gradients
    target_loss = target_output.sum()
    target_grads = torch.autograd.grad(target_loss, model.parameters(), create_graph=True)

    # Generate a random text embedding as the initial dummy input
    dummy_embedding = generate_random_text_embedding()
    optimizer = optim.Adam([dummy_embedding], lr=lr)

    for _ in range(iterations):
        optimizer.zero_grad()
        dummy_output = model(inputs_embeds=dummy_embedding).pooler_output
        dummy_loss = dummy_output.sum()
        dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        # Compute the loss between gradients (L2 + L1 distance)
        loss = sum(torch.nn.functional.mse_loss(dg, tg) + torch.nn.functional.l1_loss(dg, tg) for dg, tg in
                   zip(dummy_grads, target_grads))
        loss.backward()
        optimizer.step()

    return dummy_embedding.detach()


# Example usage
if __name__ == "__main__":
    # Define a target text
    target_text = "The quick brown fox jumps over the lazy dog."

    # Run the attack
    reconstructed_text_embedding = tag_attack(model, target_text)

    print("TAG attack completed. Reconstructed text embedding generated.")
