# MMGIA_v2.py - Fully Refactored Version

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from transformers import BertTokenizer, BertModel
import clip
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt


# ---------- Dataset & Transform ----------
class MultimodalDataset(Dataset):
    def __init__(self, dataset_name, root_dir):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self._load_data()

    def _load_data(self):
        if self.dataset_name == 'cifar100':
            self.data = datasets.CIFAR100(root=self.root_dir, download=True)
            self.classes = self.data.classes
        elif self.dataset_name == 'openi':
            self.image_dir = os.path.join(self.root_dir, 'images')
            self.text_df = pd.read_csv(os.path.join(self.root_dir, 'reports.csv'))
        elif self.dataset_name == 'flickr30k':
            self.image_dir = os.path.join(self.root_dir, 'images')
            self.captions = pd.read_csv(os.path.join(self.root_dir, 'captions.csv'))
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def __len__(self):
        if self.dataset_name == 'cifar100':
            return len(self.data)
        elif self.dataset_name == 'openi':
            return len(self.text_df)
        else:
            return len(self.captions)

    def __getitem__(self, idx):
        if self.dataset_name == 'cifar100':
            img, label = self.data[idx]
            text = f"A {self.classes[label]} in the image"
        elif self.dataset_name == 'openi':
            img_path = os.path.join(self.image_dir, self.text_df.iloc[idx]['image_name'])
            img = Image.open(img_path).convert('RGB')
            text = self.text_df.iloc[idx]['report']
        else:
            img_path = os.path.join(self.image_dir, self.captions.iloc[idx]['image_name'])
            img = Image.open(img_path).convert('RGB')
            text = self.captions.iloc[idx]['caption']

        img = self.transform(img)
        return {'image': img, 'text': text}


# ---------- Model Loader ----------
class ModelManager:
    def __init__(self, dataset_name, device='cuda', use_resnet50=True):
        self.device = device
        model_name = 'resnet50' if use_resnet50 else 'resnet50'
        self.img_model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True).to(device)
        bert_path = 'monologg/biobert_v1.1_pubmed' if dataset_name == 'openi' else 'bert-base-uncased'
        self.txt_model = BertModel.from_pretrained(bert_path).to(device)
        self.clip_model, _ = clip.load("ViT-B/32", device=device)

        for p in self.img_model.parameters(): p.requires_grad = True
        for p in self.txt_model.parameters(): p.requires_grad = True
        for p in self.clip_model.parameters(): p.requires_grad = False  # CLIP 只用于推理

    def get_models(self):
        return self.img_model, self.txt_model, self.clip_model


# ---------- Gradient Computer ----------
class GradientComputer:
    def __init__(self, img_model, txt_model, device='cuda'):
        self.img_model = img_model
        self.txt_model = txt_model
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def compute(self, batch):
        img = batch['image'].to(self.device)
        text = batch['text']

        img_out = self.img_model(img)
        img_loss = img_out.sum()
        img_grads = torch.autograd.grad(img_loss, [p for p in self.img_model.parameters() if p.requires_grad], retain_graph=True)
        img_grads = [g.detach() for g in img_grads]

        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        emb_layer = self.txt_model.get_input_embeddings()
        input_embeds = emb_layer(inputs['input_ids']).detach().clone().requires_grad_(True)
        output = self.txt_model(inputs_embeds=input_embeds)
        txt_loss = output.last_hidden_state.sum()
        txt_grads = torch.autograd.grad(txt_loss, input_embeds, retain_graph=True)
        txt_grads = [g.detach() for g in txt_grads]

        return img_grads, txt_grads, input_embeds.shape[1]


# ---------- Stage 1 Reconstructors ----------
class ImageReconstructor:
    def __init__(self, model, device='cuda', iterations=500, lr=1e-2, tv_weight=1e-2):
        self.model = model
        self.device = device
        self.iterations = iterations
        self.lr = lr
        self.tv_weight = tv_weight

    def reconstruct(self, true_grads):
        x = torch.randn(1, 3, 224, 224, device=self.device).clamp(0, 1).detach().requires_grad_(True)
        optimizer = optim.Adam([x], lr=self.lr)

        for i in range(self.iterations):
            optimizer.zero_grad()
            out = self.model(x)
            dummy_loss = out.sum()
            dummy_grads = torch.autograd.grad(dummy_loss, [p for p in self.model.parameters() if p.requires_grad], create_graph=True)
            grad_loss = sum(F.mse_loss(d, t) for d, t in zip(dummy_grads, true_grads))
            tv_loss = self.total_variation(x)
            total_loss = grad_loss + self.tv_weight * tv_loss
            total_loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"[Stage1][Image][{i}] grad_loss={grad_loss.item():.4f}, tv={tv_loss.item():.4f}")

        return x.detach()

    def total_variation(self, x):
        return (torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum() +
                torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()) / (x.size(0) * x.size(2) * x.size(3))


class TextReconstructor:
    def __init__(self, model, device='cuda', iterations=500, lr=1e-2):
        self.model = model
        self.device = device
        self.iterations = iterations
        self.lr = lr

    def reconstruct(self, true_grads, seq_len):
        x = torch.randn(1, seq_len, 768, device=self.device).detach().requires_grad_(True)
        optimizer = optim.Adam([x], lr=self.lr)

        for i in range(self.iterations):
            optimizer.zero_grad()
            out = self.model(inputs_embeds=x)
            dummy_loss = out.last_hidden_state.sum()
            dummy_grads = torch.autograd.grad(dummy_loss, x, create_graph=True)
            cos_loss = 1 - F.cosine_similarity(dummy_grads[0].flatten(), true_grads[0].flatten(), dim=0).mean()
            cos_loss.backward()
            optimizer.step()

        return x.detach()


# ---------- Stage 2 Refiner ----------
class CrossModalRefiner:
    def __init__(self, clip_model, img_model, txt_model, device='cuda', t2_iters=50, lr=1e-2, threshold=0.6):
        self.clip = clip_model
        self.img_model = img_model
        self.txt_model = txt_model
        self.device = device
        self.iterations = t2_iters
        self.lr = lr
        self.threshold = threshold

    def encode_clip(self, x_img, x_txt, decode_fn):
        z_img = self.clip.encode_image(x_img)
        decoded_text = decode_fn(x_txt)  # 仍然是非可导的部分
        with torch.no_grad():  # 仅禁止 decode 的部分
            tokenized = clip.tokenize([decoded_text]).to(self.device)
        z_txt = self.clip.encode_text(tokenized).float()
        return z_img, z_txt

    def compute_quality_score(self, model, x, true_grads, modality):
        x = x.detach().clone().requires_grad_(True)
        if modality == 'image':
            out = model(x)
            loss = out.sum()
            grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], retain_graph=True)
        else:
            out = model(inputs_embeds=x)
            loss = out.last_hidden_state.sum()
            grads = torch.autograd.grad(loss, x, retain_graph=True)
        diff = sum(torch.norm(d - t) for d, t in zip(grads, true_grads))
        norm = sum(torch.norm(t) for t in true_grads)
        return 1 - (diff / norm).item()

    def update_modality(self, x, model_type, z_fused, decode_fn=None):
        x = x.detach().clone().requires_grad_(True)
        optimizer = optim.Adam([x], lr=self.lr)

        for _ in range(5):
            optimizer.zero_grad()
            if model_type == 'image':
                z_current = self.clip.encode_image(x)
            elif model_type == 'text':
                raise RuntimeError("Cannot optimize text embedding through CLIP tokenization pipeline. Use only image modality here.")
            else:
                raise ValueError("Invalid model_type. Choose from 'image' or 'text'.")

            z_current = z_current.float()
            z_fused = z_fused.float()

            loss = F.mse_loss(z_current, z_fused)
            loss.backward()
            optimizer.step()

        return x.detach()


    def refine(self, x_img, x_txt, img_grads, txt_grads, decode_fn):
        for t in range(self.iterations):
            z_img, z_txt = self.encode_clip(x_img, x_txt, decode_fn)
            sim = F.cosine_similarity(z_img, z_txt).item()
            print(f"[Stage2][{t}] sim(z_img, z_txt)={sim:.4f}")
            if sim > self.threshold:
                print("✅ Converged")
                break
            q_img = self.compute_quality_score(self.img_model, x_img, img_grads, 'image')
            q_txt = self.compute_quality_score(self.txt_model, x_txt, txt_grads, 'text')
            w_img, w_txt = q_img / (q_img + q_txt), q_txt / (q_img + q_txt)
            z_fused = (w_img * z_img + w_txt * z_txt).detach()
            x_img = self.update_modality(x_img, model_type='image', z_fused=z_fused, decode_fn=decode_fn)
        return x_img, x_txt


# ---------- Decode Function ----------
def decode_text_from_embedding(embedding, tokenizer, embedding_weights):
    with torch.no_grad():
        seq = []
        for token_embed in embedding.squeeze(0):
            sim = F.cosine_similarity(token_embed.unsqueeze(0), embedding_weights)
            nearest_token_id = torch.argmax(sim).item()
            seq.append(nearest_token_id)
        return tokenizer.decode(seq, skip_special_tokens=True)


# ---------- Evaluation Function ----------
def evaluate(original, reconstructed):
    ori_img = original['image']
    rec_img = reconstructed['image']
    if ori_img.ndim == 4: ori_img = ori_img.squeeze(0)
    if ori_img.shape[0] == 3: ori_img = ori_img.permute(1, 2, 0)
    if rec_img.shape[0] == 3: rec_img = rec_img.permute(1, 2, 0)
    ori_img, rec_img = ori_img.cpu().numpy(), rec_img.cpu().numpy()
    if ori_img.shape != rec_img.shape:
        rec_img = resize(rec_img, ori_img.shape, preserve_range=True, anti_aliasing=True)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.imshow((ori_img * 0.5 + 0.5).clip(0, 1)); plt.title("Original")
    plt.subplot(1, 2, 2); plt.imshow((rec_img * 0.5 + 0.5).clip(0, 1)); plt.title("Reconstructed")
    plt.tight_layout(); plt.show()
    img_score = ssim(ori_img, rec_img, data_range=1.0, channel_axis=-1)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original['text'], reconstructed['text'])
    return {'SSIM': img_score, 'ROUGE-1': scores['rouge1'].fmeasure, 'ROUGE-2': scores['rouge2'].fmeasure, 'ROUGE-L': scores['rougeL'].fmeasure}


# ---------- Main Execution ----------
if __name__ == '__main__':
    dataset_name = 'cifar100'
    root_dir = f'./data/{dataset_name}'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = MultimodalDataset(dataset_name, root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    sample = next(iter(dataloader))
    print(f"\nOriginal text: {sample['text']}")

    model_mgr = ModelManager(dataset_name, device=device)
    img_model, txt_model, clip_model = model_mgr.get_models()
    gradient_computer = GradientComputer(img_model, txt_model, device=device)
    img_grads, txt_grads, seq_len = gradient_computer.compute(sample)

    x_img = ImageReconstructor(img_model, device=device, iterations=1000, tv_weight=1e-3).reconstruct(img_grads)
    x_txt = TextReconstructor(txt_model, device=device).reconstruct(txt_grads, seq_len)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    embedding_weights = txt_model.get_input_embeddings().weight
    decode_fn = lambda x: decode_text_from_embedding(x, tokenizer, embedding_weights)

    refiner = CrossModalRefiner(clip_model, img_model, txt_model, device=device)
    x_img, x_txt = refiner.refine(x_img, x_txt, img_grads, txt_grads, decode_fn)
    decoded_text = decode_text_from_embedding(x_txt, tokenizer, embedding_weights)

    evaluation = {
        'original': {
            'image': sample['image'].squeeze(0),
            'text': sample['text'][0]  # ✅ 取出字符串
        },
        'reconstructed': {
            'image': x_img.squeeze(0).cpu(),
            'text': decoded_text
        }
    }

    metrics = evaluate(evaluation['original'], evaluation['reconstructed'])
    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
