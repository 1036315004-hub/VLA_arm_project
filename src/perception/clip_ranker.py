# src/perception/clip_ranker.py
import open_clip
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import List, Dict

class CLIPRanker:
    def __init__(self, model_name="ViT-B-32", device="cpu"):
        # open_clip: model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s32b_b79k')
        self.device = device
        # 选择较小的预训练模型以减少资源
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s32b_b79k')
        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, texts: List[str]):
        # 返回 numpy float32 vectors (n_texts, dim)
        with torch.no_grad():
            tokens = open_clip.tokenize(texts).to(self.device)
            txt_feats = self.model.encode_text(tokens)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            return txt_feats.cpu().numpy()

    def encode_image(self, crops: List[np.ndarray]):
        """
        crops: list of HxWx3 RGB uint8 numpy arrays
        返回 (n_crops, dim)
        """
        imgs = [self.preprocess(Image=crop) if False else self.preprocess(Image.fromarray(crop)) for crop in crops]
        imgs = torch.stack(imgs).to(self.device)
        with torch.no_grad():
            img_feats = self.model.encode_image(imgs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            return img_feats.cpu().numpy()

    def score_crops(self, crops: List[np.ndarray], query: str) -> List[float]:
        # 为了节省内存，编码文本一次，再编码图片分batch
        if len(crops) == 0:
            return []
        txt_vec = self.encode_text([query])[0]  # (dim,)
        # batch encoding
        batch_size = 16
        scores = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i+batch_size]
            imgs = [self.preprocess(Image.fromarray(c)) for c in batch]
            imgs = torch.stack(imgs).to(self.device)
            with torch.no_grad():
                img_feats = self.model.encode_image(imgs)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                img_feats = img_feats.cpu().numpy()
            # cosine similarity
            for v in img_feats:
                scores.append(float(np.dot(v, txt_vec)))
        return scores