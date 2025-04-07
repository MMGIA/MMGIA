import torch
import torch.nn as nn
import torch.optim as optim
import clip  # OpenAI CLIP
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import BertTokenizer, BertModel

# Hyperparameters
T1_ITERATIONS = 100  # Stage 1 iterations
T2_ITERATIONS = 50   # Stage 2 iterations
THETA = 0.7         # Convergence threshold
LR_STAGE1 = 0.1     # Learning rate for stage 1
LR_STAGE2 = 0.01    # Learning rate for stage 2
TV_WEIGHT = 1e-6    # Total variation regularization weight

class MMGIA:
    def __init__(self, img_model, txt_model, clip_model):
        """
        Initialize MMGIA with target models and CLIP for cross-modal alignment
        
        Args:
            img_model: Target image model (e.g., ResNet)
            txt_model: Target text model (e.g., BERT)
            clip_model: Pre-trained CLIP model for cross-modal alignment
        """
        self.img_model = img_model
        self.txt_model = txt_model
        self.clip_model = clip_model
        
        # Freeze target models (we only need their gradients)
        for param in self.img_model.parameters():
            param.requires_grad = False
        for param in self.txt_model.parameters():
            param.requires_grad = False
            
        # Initialize CLIP components
        self.clip_img_encoder = clip_model.visual
        self.clip_txt_encoder = clip_model.transformer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def reconstruct(self, img_grads, txt_grads):
        """
        Main reconstruction method implementing the two-stage MMGIA algorithm
        
        Args:
            img_grads: Gradients from image model
            txt_grads: Gradients from text model
            
        Returns:
            Tuple of (reconstructed_image, reconstructed_text)
        """
        # Stage 1: Modality-specific reconstruction
        x_img = self.stage1_img_reconstruction(img_grads)
        x_txt = self.stage1_txt_reconstruction(txt_grads)
        
        # Stage 2: Correlation-driven refinement
        x_img, x_txt = self.stage2_refinement(x_img, x_txt, img_grads, txt_grads)
        
        return x_img, x_txt
    
    def stage1_img_reconstruction(self, true_grads):
        """
        Stage 1: Image modality reconstruction using gradient matching
        """
        # Initialize random image (3x224x224)
        x_img = torch.randn(1, 3, 224, 224, requires_grad=True)
        optimizer = optim.Adam([x_img], lr=LR_STAGE1)
        
        for _ in range(T1_ITERATIONS):
            optimizer.zero_grad()
            
            # Forward pass through image model
            dummy_output = self.img_model(x_img)
            dummy_loss = dummy_output.sum()  # Dummy loss
            dummy_grads = torch.autograd.grad(dummy_loss, self.img_model.parameters(), 
                                            create_graph=True)
            
            # Compute gradient matching loss
            grad_loss = 0
            for (dummy_p, true_p) in zip(dummy_grads, true_grads):
                grad_loss += torch.norm(dummy_p - true_p, p=2)
                
            # Add TV regularization
            tv_loss = self.total_variation(x_img)
            
            total_loss = grad_loss + TV_WEIGHT * tv_loss
            total_loss.backward()
            optimizer.step()
            
        return x_img.detach()
    
    def stage1_txt_reconstruction(self, true_grads):
        """
        Stage 1: Text modality reconstruction using cosine similarity
        """
        # Initialize random text embeddings (batch_size=1, seq_len=32, hidden_dim=768)
        x_txt = torch.randn(1, 32, 768, requires_grad=True)
        optimizer = optim.Adam([x_txt], lr=LR_STAGE1)
        
        for _ in range(T1_ITERATIONS):
            optimizer.zero_grad()
            
            # Forward pass through text model
            dummy_output = self.txt_model(inputs_embeds=x_img)
            dummy_loss = dummy_output.last_hidden_state.sum()  # Dummy loss
            dummy_grads = torch.autograd.grad(dummy_loss, self.txt_model.parameters(), 
                                            create_graph=True)
            
            # Compute cosine similarity loss
            cos_loss = 0
            for (dummy_p, true_p) in zip(dummy_grads, true_grads):
                cos_loss += 1 - torch.cosine_similarity(dummy_p.flatten(), 
                                                      true_p.flatten(), dim=0)
                
            total_loss = cos_loss
            total_loss.backward()
            optimizer.step()
            
        return x_txt.detach()
    
    def stage2_refinement(self, x_img, x_txt, img_grads, txt_grads):
        """
        Stage 2: Correlation-driven refinement using CLIP embeddings
        """
        for _ in range(T2_ITERATIONS):
            # Get CLIP embeddings for both modalities
            z_img = self.clip_img_encoder(x_img)
            z_txt = self.clip_txt_encoder(x_txt)
            
            # Check convergence
            sim = torch.cosine_similarity(z_img, z_txt, dim=-1)
            if sim > THETA:
                break
                
            # Compute quality scores for each modality
            q_img = self.compute_quality_score(x_img, img_grads, modality='image')
            q_txt = self.compute_quality_score(x_txt, txt_grads, modality='text')
            
            # Compute dynamic weights
            w_img = q_img / (q_img + q_txt)
            w_txt = q_txt / (q_img + q_txt)
            
            # Compute fused embedding
            z_fused = w_img * z_img + w_txt * z_txt
            
            # Update each modality to align with fused embedding
            x_img = self.update_modality(x_img, z_img, z_fused, modality='image')
            x_txt = self.update_modality(x_txt, z_txt, z_fused, modality='text')
            
        return x_img, x_txt
    
    def compute_quality_score(self, x, true_grads, modality):
        """
        Compute quality score for a modality (Eq. 6 in paper)
        """
        if modality == 'image':
            dummy_output = self.img_model(x)
            dummy_loss = dummy_output.sum()
            dummy_grads = torch.autograd.grad(dummy_loss, self.img_model.parameters(), 
                                            create_graph=True)
        else:  # text
            dummy_output = self.txt_model(inputs_embeds=x)
            dummy_loss = dummy_output.last_hidden_state.sum()
            dummy_grads = torch.autograd.grad(dummy_loss, self.txt_model.parameters(), 
                                            create_graph=True)
        
        grad_diff = 0
        true_grad_norm = 0
        for (dummy_p, true_p) in zip(dummy_grads, true_grads):
            grad_diff += torch.norm(dummy_p - true_p, p=2)
            true_grad_norm += torch.norm(true_p, p=2)
            
        return 1 - (grad_diff / true_grad_norm)
    
    def update_modality(self, x, z, z_fused, modality):
        """
        Update modality reconstruction to align with fused embedding
        """
        x = x.clone().requires_grad_(True)
        optimizer = optim.Adam([x], lr=LR_STAGE2)
        
        for _ in range(5):  # Few inner iterations
            optimizer.zero_grad()
            
            if modality == 'image':
                z_current = self.clip_img_encoder(x)
            else:  # text
                z_current = self.clip_txt_encoder(x)
                
            loss = torch.norm(z_current - z_fused, p=2)
            loss.backward()
            optimizer.step()
            
        return x.detach()
    
    def total_variation(self, img):
        """
        Total variation regularization for image smoothness
        """
        batch_size = img.size(0)
        height = img.size(2)
        width = img.size(3)

        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
        tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()

        return (tv_h + tv_w) / (batch_size * height * width)

# Example usage
if __name__ == "__main__":
    # Load models (in practice, these would be the target models)
    img_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    txt_model = BertModel.from_pretrained('bert-base-uncased')
    clip_model, _ = clip.load("ViT-B/32")
    
    # Initialize MMGIA
    attacker = MMGIA(img_model, txt_model, clip_model)
    
    # Example gradients (in practice, these would be intercepted from FL)
    img_grads = [torch.randn_like(p) for p in img_model.parameters()]
    txt_grads = [torch.randn_like(p) for p in txt_model.parameters()]
    
    # Run reconstruction
    rec_img, rec_txt = attacker.reconstruct(img_grads, txt_grads)
    
    print("Reconstruction complete!")
    print(f"Image shape: {rec_img.shape}")
    print(f"Text embedding shape: {rec_txt.shape}")
