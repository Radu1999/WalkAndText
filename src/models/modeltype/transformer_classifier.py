import numpy as np
import torch
import torch.nn as nn
from ..architectures.transformer import ProjectionHead
from ..tools.losses import get_loss_function
from ..rotation2xyz import Rotation2xyz
import torch.nn.functional as F
from tqdm import tqdm
from src.utils.action_label_to_idx import action_label_to_idx
import random

class PoseClassifier(nn.Module):
    def __init__(self, encoder, device, lambdas, latent_dim, outputxyz,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, num_labels=60, hidden_size=768, clip_lambdas={}, **kwargs):
        super().__init__()

        self.encoder = encoder
        
        self.outputxyz = outputxyz

        self.lambdas = lambdas
        self.clip_lambdas = clip_lambdas

        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.device = device
        self.translation = translation
        self.jointstype = jointstype
        self.vertstrans = vertstrans
        self.rotation2xyz = Rotation2xyz(device=self.device)
        self.num_labels = num_labels
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )

        
        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans}
        self.loss = nn.CrossEntropyLoss()

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
        # encode
        motion_embeddings = self.encoder(batch)["mu"]
        motion_embeddings = F.normalize(motion_embeddings, p=2, dim=1)
        return motion_embeddings
        #return self.motion_projection(motion_embeddings)
    
    def predict(self, batch):
        motion_features = self.encode_motion(batch)
        classification_output = self.classifier(motion_features)
        return classification_output
    
    def forward(self, batch):
        # Get labels
        possible_labels = list(map(lambda x: [action_label_to_idx[cat] for cat in x], batch['all_categories']))
        selected_labels = [random.choice(possible_label) for possible_label in possible_labels]        
        classification_output = self.predict(batch)
        
        bs = batch['x'].shape[0]
        labels = torch.zeros((bs, self.num_labels)).to(batch['x'].device)
        labels[range(bs), selected_labels] = 1.0
        
        loss = self.loss(classification_output, labels)  
        losses = {"loss": loss.item()}
        
        return loss, losses
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
    
