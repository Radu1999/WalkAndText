import numpy as np
import torch
import torch.nn as nn
from transformers import MPNetTokenizerFast, AutoModel
import clip
from ..architectures.transformer import ProjectionHead
from ..tools.losses import get_loss_function
from ..rotation2xyz import Rotation2xyz
import torch.nn.functional as F
from angle_emb import AnglE
from tqdm import tqdm
import random
import joblib

loss_motion = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, text_features, motion_features):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp().to(motion_features.device)
        logits_per_motion = logit_scale * motion_features @ text_features.t()
        logits_per_text = logits_per_motion.t()
        
        ground_truth = torch.arange(len(motion_features),dtype=torch.long,device=motion_features.device)

        total_loss = (loss_motion(logits_per_motion,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        return total_loss

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    
class CLIPose(nn.Module):
    def __init__(self, encoder, device, lambdas, latent_dim, outputxyz,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, text_sources=None, clip_lambdas={}, **kwargs):
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
        self.text_sources = text_sources
        
        self.text_projection = ProjectionHead(embedding_dim=768, projection_dim=512)
        #self.motion_projection = ProjectionHead(embedding_dim=512, projection_dim=256)
        
        model_name = 'sentence-transformers/all-mpnet-base-v2'
        self.tokenizer = MPNetTokenizerFast.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # self.losses = list(self.lambdas) + ["mixed"]
        # self.text_encoder = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

        self.rotation2xyz = Rotation2xyz(device=self.device)
        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans}
        self.loss = CLIPLoss()
        # Initialize weights
        # self._init_weights()

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
    
    def encode_text(self, text):
        # return self.text_projection(self.text_encoder.encode(text, to_numpy=False))
        encoder_input = self.tokenizer(text, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        encoder_output = self.text_encoder(**encoder_input)
        sentence_embeddings = mean_pooling(encoder_output, encoder_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        # return sentence_embeddings
        return self.text_projection(sentence_embeddings)
    
    def forward(self, batch):
        if self.text_sources is None:
            try:
                text_features = self.encode_text(batch['clip_text'])
            except:
                print(batch['clip_text'])
                exit(0)
        else:
            # choose category
            categories = [random.choice(inner_list) for inner_list in batch['all_categories']]
            descriptions = [random.choice(self.text_sources[category]) for category in categories]
            text_features = self.encode_text(descriptions)

        motion_features = self.encode_motion(batch)
        loss = self.loss(text_features, motion_features)  
        losses = {"loss": loss.item()}
        batch["motion_features"] = motion_features
        
        return loss, losses
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
    
