import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import clip
from ..architectures.transformer import ProjectionHead
from ..tools.losses import get_loss_function
from ..rotation2xyz import Rotation2xyz
import torch.nn.functional as F
from tqdm import tqdm
import random
import joblib
from angle_emb import AnglE
from transformers import AutoModel, AutoTokenizer

cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
l2gen = joblib.load('label_mapping.pt')
triplet_loss = nn.TripletMarginLoss(margin=10.0)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, text_features, motion_features, feature_ids):
        unique_ids = list(set(feature_ids))  # Get unique string ids
        id_to_int = {id_str: idx for idx, id_str in enumerate(unique_ids)}  # Map strings to integers
        feature_ids_int = torch.tensor([id_to_int[fid] for fid in feature_ids], device=text_features.device)
        
        perm_indices = torch.randperm(text_features.size(0), device=text_features.device)

        feature_ids_int = feature_ids_int.view(-1)  # Ensure it's 1D (batch_size,)
        permuted_ids = feature_ids_int[perm_indices].view(-1)  # Also 1D (batch_size,)

        mask = (feature_ids_int == permuted_ids)  # Shape: (batch_size,)

        mask = mask.unsqueeze(1)  # Shape: (batch_size, 1), matches (batch_size, feature_dim)
        negative_features = torch.where(mask, motion_features, text_features[perm_indices])

        loss = triplet_loss(motion_features, text_features, negative_features)

        return loss


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    
class WalkText(nn.Module):
    def __init__(self, encoder, device, latent_dim,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, outputxyz=False, text_sources=None, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.motion_projection = ProjectionHead(embedding_dim=768, projection_dim=1024)
        self._init_weights()

        self.outputxyz = outputxyz

        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.device = device
        self.translation = translation
        self.jointstype = jointstype
        self.vertstrans = vertstrans
        self.text_sources = text_sources
        self.text_encoder = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls')
        self.text_encoder.device = 'cuda:0'
        self.text_encoder = self.text_encoder.cuda()
        
        self.rotation2xyz = Rotation2xyz(device=self.device)
        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans}
        self.loss = ContrastiveLoss()
        

    def rot2xyz(self, x, mask, get_rotations_back=False, **kwargs):
        kargs = self.param2xyz.copy()
        kargs.update(kwargs)
        return self.rotation2xyz(x, mask, get_rotations_back=get_rotations_back, **kargs)

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask
    
    def encode_motion(self, batch):
        if self.outputxyz:
            batch["x_xyz"] = self.rot2xyz(batch["x"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]
        motion_embeddings = self.encoder(batch)["mu"]
        batch['motion_embeddings'] = motion_embeddings
        return self.motion_projection(motion_embeddings)
    
    def encode_text(self, text):
        return self.text_encoder.encode(text, to_numpy=False)
    
    def forward(self, batch):
        motion_features = self.encode_motion(batch)
        text_features =  self.encode_text(batch['y'])
        feature_ids = batch['id']
        
        loss = self.loss(text_features, motion_features, feature_ids)  
        losses = {"loss": loss.item()}
        batch["motion_features"] = motion_features   
        return loss, losses
    
            
    
    def freeze_weights(self):
        for p in self.parameters():
            p.requires_grad = False 
                
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)