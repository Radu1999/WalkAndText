import torch
from tqdm import tqdm
from copy import deepcopy
import pickle
import clip
import wandb
import torch.nn.functional as F
import numpy as np
from pytorch_metric_learning import losses
import torch.nn as nn

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss
    
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

loss_motion = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, motion_features, text_features):

        # normalized features
        motion_features = motion_features / motion_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp().to(motion_features.device)
        logits_per_motion = logit_scale * motion_features @ text_features.t()
        logits_per_text = logits_per_motion.t()
        
        ground_truth = torch.arange(len(motion_features),dtype=torch.long,device=motion_features.device)

        total_loss = (loss_motion(logits_per_motion,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        return total_loss

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def train_or_test(model, optimizer, iterator, device, mode="train"):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")
        


    # loss of the epoch
    clip_loss = CLIPLoss()
    dict_loss = {}
    model_reps = []
    text_reps = []
    image_reps = []
    batch_size = 80
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    use_cache = True
    with grad_env():
        # Get repr for gradient cache
        if use_cache:
            with torch.no_grad():
                for i, batch in tqdm(enumerate(iterator), desc="Computing batch for caching"):
                    # Put everything in device
                    # Added if is_tensor as 'clip_text' in batch is a list of strings, not a tensor!
                    batch = {key: val.to(device) if torch.is_tensor(val) else val for key, val in batch.items()}
                    # forward pass
                    batch = model(batch)
                    batch_size = batch["z"].shape[0]
                    model_reps.append(batch["z"])
                    text_rep = model.encode_text(batch['clip_text'], method="clip")
                    text_reps.append(text_rep)
                    # image_rep = model.encode_image(batch['clip_images'], method="clip")
                    # image_reps.append(image_rep)

            model_reps = torch.cat(model_reps, dim=0)
            text_reps = torch.cat(text_reps, dim=0)
            #image_reps = torch.cat(image_reps, dim=0)

            # model_reps = F.normalize(model_reps, p=2, dim=-1)
            # text_reps = F.normalize(text_reps, p=2, dim=-1)
            # Calculate loss
            model_reps.requires_grad_()
            
            loss_text = clip_loss(model_reps, text_reps)
            # target = torch.eye(model_reps.shape[0]).to(device)
            # contrastive = SimCLR_Loss(model_reps.shape[0], 0.07)
            # loss_text = contrastive(model_reps, text_reps, target)
             
            wandb.log({'loss_text': loss_text})
            # loss_image = contrastive(model_reps, image_reps, target)
            # wandb.log({'loss_image': loss_image})
            loss = loss_text # + loss_image

            loss.backward()

            # Create cache
            cache = model_reps.grad
            loss = loss.detach()
        
        
        counter = 0
        start_idx = 0
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            # Put everything in device
            # Added if is_tensor as 'clip_text' in batch is a list of strings, not a tensor!
            batch = {key: val.to(device) if torch.is_tensor(val) else val for key, val in batch.items()}
            counter += 1

            if not use_cache:
                # update the gradients to zero
                optimizer.zero_grad()

            # forward pass
            batch = model(batch)

            # mixed_loss, losses = model.compute_clip_losses(batch)

            # if i == 0:
            #     dict_loss = deepcopy(losses)
            # else:
            #     for key in dict_loss.keys():
            #         if (i + 1) % 51 == 0:
            #             wandb.log({key: losses[key]})
            #         dict_loss[key] += losses[key]

            if mode == "train":
                # backward pass
                if use_cache:
                    gradient = cache[start_idx:start_idx + batch['z'].shape[0]]
                # mixed_loss.backward(retain_graph=use_cache)
                if use_cache:
                    batch['z'].backward(gradient, retain_graph=False)
                # update the weights
                if model.clip_training:
                    convert_models_to_fp32(model.clip_model)
                    if not use_cache:
                        optimizer.step()
                        clip.model.convert_weights(model.clip_model)
                    
                else:
                    if not use_cache:
                        optimizer.step()
        
        if use_cache:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= counter
            optimizer.step()
            clip.model.convert_weights(model.clip_model)
        
    torch.cuda.empty_cache()
    return dict_loss


def train(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="train")


def test(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="test")
