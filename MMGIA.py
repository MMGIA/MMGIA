import torch
import torch.nn as nn
import torch.optim as optim
import clip
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
from rouge_score import rouge_scorer
import numpy as np


class MultimodalDataset(Dataset):
    def __init__(self, dataset_name, root_dir, transform=None):
        self.dataset_name = dataset_name
        self.transform = transform
        
        if dataset_name == 'cifar100':
            self.dataset = datasets.CIFAR100(root=root_dir, download=True)
            self.classes = self.dataset.classes
        elif dataset_name == 'openi':
            self.image_dir = os.path.join(root_dir, 'images')
            self.text_df = pd.read_csv(os.path.join(root_dir, 'reports.csv'))
        elif dataset_name == 'flickr30k':
            self.image_dir = os.path.join(root_dir, 'images')
            self.captions = pd.read_csv(os.path.join(root_dir, 'captions.csv'))
        else:
            raise ValueError("Unsupported dataset")

    def __len__(self):
        if self.dataset_name == 'cifar100':
            return len(self.dataset)
        elif self.dataset_name == 'openi':
            return len(self.text_df)
        else:  # flickr30k
            return len(self.captions)

    def __getitem__(self, idx):
        if self.dataset_name == 'cifar100':
            img, label = self.dataset[idx]
            text = f"A {self.classes[label]} in the image"
            if self.transform:
                img = self.transform(img)
            return {'image': img, 'text': text, 'label': label}
            
        elif self.dataset_name == 'openi':
            img_path = os.path.join(self.image_dir, self.text_df.iloc[idx]['image_name'])
            img = Image.open(img_path).convert('RGB')
            text = self.text_df.iloc[idx]['report']
            if self.transform:
                img = self.transform(img)
            return {'image': img, 'text': text}
            
        else:  # flickr30k
            img_path = os.path.join(self.image_dir, self.captions.iloc[idx]['image_name'])
            img = Image.open(img_path).convert('RGB')
            text = self.captions.iloc[idx]['caption']
            if self.transform:
                img = self.transform(img)
            return {'image': img, 'text': text}

def get_transforms(dataset_name):
    if dataset_name == 'cifar100':
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name == 'openi':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # flickr30k
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class MMGIA:
    def __init__(self, dataset_name, root_dir, device='cuda'):
        self.device = device
        self.dataset_name = dataset_name
        self.transform = get_transforms(dataset_name)
        self.dataset = MultimodalDataset(dataset_name, root_dir, self.transform)
        
      
        self.init_models()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
     
        self.T1_ITERATIONS = 100
        self.T2_ITERATIONS = 50
        self.THRESHOLD = 0.7
        self.LR_STAGE1 = 0.1
        self.LR_STAGE2 = 0.01
        self.TV_WEIGHT = 1e-6
        
    def init_models(self):
        if self.dataset_name == 'openi':
            print("Initializing models for OpenI dataset...")
            self.img_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50').to(self.device)
            self.txt_model = BertModel.from_pretrained("monologg/biobert_v1.1_pubmed").to(self.device)
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        else:
            self.img_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50').to(self.device)
            self.txt_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        
        for param in self.img_model.parameters():
            param.requires_grad = False
        for param in self.txt_model.parameters():
            param.requires_grad = False

    def compute_gradients(self, batch):
        img = batch['image'].unsqueeze(0).to(self.device)
        img_output = self.img_model(img)
        img_loss = img_output.sum()
        img_grads = torch.autograd.grad(img_loss, self.img_model.parameters(), retain_graph=True)
        
        text = batch['text']
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        txt_output = self.txt_model(**inputs).last_hidden_state
        txt_loss = txt_output.sum()
        txt_grads = torch.autograd.grad(txt_loss, self.txt_model.parameters())
        
        return img_grads, txt_grads

    def reconstruct(self, img_grads, txt_grads):
        x_img = self.stage1_img_reconstruction(img_grads)
        x_txt = self.stage1_txt_reconstruction(txt_grads)
        
        x_img, x_txt = self.stage2_refinement(x_img, x_txt, img_grads, txt_grads)
        
        return x_img, x_txt

    def stage1_img_reconstruction(self, true_grads):
        x_img = torch.randn(1, 3, 224, 224, requires_grad=True, device=self.device)
        optimizer = optim.Adam([x_img], lr=self.LR_STAGE1)
        
        for _ in range(self.T1_ITERATIONS):
            optimizer.zero_grad()
            
            dummy_output = self.img_model(x_img)
            dummy_loss = dummy_output.sum()
            dummy_grads = torch.autograd.grad(dummy_loss, self.img_model.parameters(), 
                                            create_graph=True)
            
            grad_loss = sum(torch.norm(dummy_p - true_p, p=2) 
                          for dummy_p, true_p in zip(dummy_grads, true_grads))
            
            tv_loss = self.total_variation(x_img)
            
            total_loss = grad_loss + self.TV_WEIGHT * tv_loss
            total_loss.backward()
            optimizer.step()
            
        return x_img.detach()

    def stage1_txt_reconstruction(self, true_grads):
        x_txt = torch.randn(1, 32, 768, requires_grad=True, device=self.device)
        optimizer = optim.Adam([x_txt], lr=self.LR_STAGE1)
        
        for _ in range(self.T1_ITERATIONS):
            optimizer.zero_grad()
            
            dummy_output = self.txt_model(inputs_embeds=x_txt)
            dummy_loss = dummy_output.last_hidden_state.sum()
            dummy_grads = torch.autograd.grad(dummy_loss, self.txt_model.parameters(), 
                                            create_graph=True)
            
            cos_loss = sum(1 - torch.cosine_similarity(dummy_p.flatten(), true_p.flatten(), dim=0)
                         for dummy_p, true_p in zip(dummy_grads, true_grads))
            
            cos_loss.backward()
            optimizer.step()
            
        return x_txt.detach()

    def stage2_refinement(self, x_img, x_txt, img_grads, txt_grads):
        for _ in range(self.T2_ITERATIONS):
            z_img = self.clip_model.encode_image(x_img)
            z_txt = self.clip_model.encode_text(
                clip.tokenize(["dummy text"]).to(self.device)).float()
            
            sim = torch.cosine_similarity(z_img, z_txt, dim=-1)
            if sim > self.THRESHOLD:
                break
                
            q_img = self.compute_quality_score(x_img, img_grads, 'image')
            q_txt = self.compute_quality_score(x_txt, txt_grads, 'text')
            
            w_img = q_img / (q_img + q_txt)
            w_txt = q_txt / (q_img + q_txt)
            
            z_fused = w_img * z_img + w_txt * z_txt
            
            x_img = self.update_modality(x_img, z_img, z_fused, 'image')
            x_txt = self.update_modality(x_txt, z_txt, z_fused, 'text')
            
        return x_img, x_txt

    def compute_quality_score(self, x, true_grads, modality):
        if modality == 'image':
            dummy_output = self.img_model(x)
            dummy_loss = dummy_output.sum()
            dummy_grads = torch.autograd.grad(dummy_loss, self.img_model.parameters(), 
                                            retain_graph=True)
        else:  # text
            dummy_output = self.txt_model(inputs_embeds=x)
            dummy_loss = dummy_output.last_hidden_state.sum()
            dummy_grads = torch.autograd.grad(dummy_loss, self.txt_model.parameters())
        
        grad_diff = sum(torch.norm(dummy_p - true_p, p=2) 
                       for dummy_p, true_p in zip(dummy_grads, true_grads))
        true_grad_norm = sum(torch.norm(true_p, p=2) for true_p in true_grads)
        
        return 1 - (grad_diff / true_grad_norm).item()

    def update_modality(self, x, z, z_fused, modality):
        x = x.clone().requires_grad_(True)
        optimizer = optim.Adam([x], lr=self.LR_STAGE2)
        
        for _ in range(5):  
            optimizer.zero_grad()
            
            if modality == 'image':
                z_current = self.clip_model.encode_image(x)
            else:  # text
                z_current = self.clip_model.encode_text(
                    clip.tokenize(["dummy text"]).to(self.device)).float()
                
            loss = torch.norm(z_current - z_fused, p=2)
            loss.backward()
            optimizer.step()
            
        return x.detach()

    def total_variation(self, img):
        batch_size, height, width = img.size(0), img.size(2), img.size(3)
        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
        tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
        return (tv_h + tv_w) / (batch_size * height * width)

    def evaluate(self, original, reconstructed):
        img_score = ssim(original['image'].cpu().numpy().transpose(1,2,0),
                        reconstructed['image'].cpu().numpy().transpose(1,2,0),
                        multichannel=True, data_range=1.0)
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        text_scores = scorer.score(original['text'], reconstructed['text'])
        
        return {
            'SSIM': img_score,
            'ROUGE-1': text_scores['rouge1'].fmeasure,
            'ROUGE-2': text_scores['rouge2'].fmeasure,
            'ROUGE-L': text_scores['rougeL'].fmeasure
        }

if __name__ == "__main__":
    DATASET_NAME = 'cifar100'  # cifar100, openi, flickr30k
    DATA_ROOT = f'./data/{DATASET_NAME}'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    attacker = MMGIA(DATASET_NAME, DATA_ROOT, device=DEVICE)
    dataloader = DataLoader(attacker.dataset, batch_size=1, shuffle=True)
    
    sample = next(iter(dataloader))
    print(f"\nOriginal text: {sample['text'][0]}")
    
    img_grads, txt_grads = attacker.compute_gradients(sample)
    
    rec_img, rec_txt = attacker.reconstruct(img_grads, txt_grads)
    evaluation = {
        'original': sample,
        'reconstructed': {
            'image': rec_img.squeeze(0).cpu(),
            'text': sample['text'][0]  
        }
    }
    metrics = attacker.evaluate(evaluation['original'], evaluation['reconstructed'])
    
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
