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
from src.utils.action_label_to_idx import action_label_to_idx
import random
import joblib

loss_motion = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
mse_loss = nn.MSELoss()
l2gen = joblib.load('label_mapping.pt')
#subset = joblib.load('inference_labels.pt')

class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.threshold = 0.9
        
    def forward(self, text_features, motion_features, gen_features):
        loss_remake_cap =  F.mse_loss(gen_features, motion_features, reduction='mean')
        loss_remake_label = F.mse_loss(text_features, motion_features, reduction='mean')
        return (loss_remake_cap + loss_remake_label) / 2
        loss_remake = F.mse_loss(gen_features, motion_features)
#         gen_features_norm = F.normalize(gen_features, p=2, dim=1)
#         text_features_norm = F.normalize(text_features, p=2, dim=1)
#         motion_features_norm = F.normalize(motion_features, p=2, dim=1)
        
#         logit_scale = self.logit_scale.exp().to(motion_features_norm.device)
#         logits_per_motion = logit_scale * motion_features_norm @ text_features_norm.t()
#         logits_per_text = logits_per_motion.t()
        
#         intrinsic_sim = text_features_norm @ text_features_norm.t()
#         target = torch.eye(len(motion_features)).to(motion_features.device)
#         mask = torch.where(intrinsic_sim >= self.threshold, torch.tensor(0.0), torch.tensor(1.0)).to(motion_features.device)
#         mask = mask + target
#         contrastive_loss = (loss_motion(logits_per_motion * mask,target) + loss_txt(logits_per_text * mask,target)) / 2
#         logits_per_motion = logit_scale * motion_features_norm @ gen_features_norm.t()
#         logits_per_text = logits_per_motion.t()
#         gen_loss = (loss_motion(logits_per_motion * mask,target) + loss_txt(logits_per_text * mask,target)) / 2
#         loss_remake =  F.mse_loss(text_features, motion_features, reduction='mean')
#         return contrastive_loss + gen_loss # + loss_remake

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    
class CLIPose(nn.Module):
    def __init__(self, encoder, decoder, device, lambdas, latent_dim, outputxyz,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, use_mlp=False, text_sources=None, clip_lambdas={}, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = None
        # self.text_projection = ProjectionHead(embedding_dim=1024, projection_dim=768, dropout=0.2)
        self.motion_projection = ProjectionHead(embedding_dim=768, projection_dim=1024)
        self._init_weights()

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
        
        
        
        
        model_name = 'WhereIsAI/UAE-Large-V1' # 'sentence-transformers/all-mpnet-base-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # self.losses = list(self.lambdas) + ["mixed"]
        
        self.rotation2xyz = Rotation2xyz(device=self.device)
        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans}
        self.loss = CLIPLoss()
        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 60))
        

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
    
    def decode_motion(self, batch):
        decoded = self.decoder(batch)["output"]
        return decoded
    
    def set_mlp(self):
        self.use_mlp = True
        #self.mapping = {label: idx for (idx, label) in enumerate(subset)}
        self.mlp = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 60))
        self.mlp.to(self.device)
    
    def encode_text(self, text):
        encoder_input = self.tokenizer(text, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        encoder_output = self.text_encoder(**encoder_input)
        sentence_embeddings = mean_pooling(encoder_output, encoder_input['attention_mask'])
        return sentence_embeddings
    
    def forward(self, batch):
        
        labels = [random.choice(labels) for labels in batch['all_categories']]
        motion_features = self.encode_motion(batch)
        if self.use_mlp:
            logits = self.mlp(motion_features)
            label_ids = torch.tensor([action_label_to_idx[label] for label in labels]).to(motion_features.device)
            loss1 = loss_motion(logits, label_ids)
            text_features = torch.stack([self.precomputed_labels[label] for label in labels])
            gen_features = torch.stack([self.precomputed[label] for label in labels])

            # batch["mask"] = (torch.rand(batch["motion_embeddings"].shape) > 0.7).to(motion_features.device)
            # decoded_features = self.decode_motion(batch)

            loss2 = self.loss(text_features, motion_features, gen_features)
            losses = {"loss": loss1.item() + 0.4 * loss2.item()}
            return loss1 + 0.4 * loss2, losses
            
        text_features =  torch.stack([self.precomputed_labels[label] for label in labels])
        gen_features = torch.stack([self.precomputed[label] for label in labels])
       
        # batch["mask"] = (torch.rand(batch["motion_embeddings"].shape) > 0.7).to(motion_features.device)
        # decoded_features = self.decode_motion(batch)
        
        loss = self.loss(text_features, motion_features, gen_features)  
        losses = {"loss": loss.item()}
        batch["motion_features"] = motion_features
        
        
        return loss, losses
    
    def predict(self, batch):
        motion_features = self.encode_motion(batch)
        classification_output = self.mlp(motion_features)
        return classification_output
    
    def precompute_tokens(self):
        self.precomputed = {}
        self.precomputed_labels = {}
        for label in list(action_label_to_idx.keys())[:60]:
            t1 = self.encode_text(l2gen[label])
            t2 = self.encode_text(label)
            self.precomputed[label] = torch.mean(t1, dim=0) 
            self.precomputed_labels[label] = torch.mean(t2, dim=0) 
    
    def freeze_weights(self):
        for p in self.parameters():
            p.requires_grad = False 
                
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)