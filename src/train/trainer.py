import torch
from tqdm import tqdm
from copy import deepcopy
import pickle
import clip
import wandb
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

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
    use_cache = False
    with grad_env():
        # Get repr for gradient cache
        if use_cache and mode == "train":
            with torch.no_grad():
                for i, batch in tqdm(enumerate(iterator), desc="Computing batch for caching"):
                    # Put everything in device
                    # Added if is_tensor as 'clip_text' in batch is a list of strings, not a tensor!
                    batch = {key: val.to(device) if torch.is_tensor(val) else val for key, val in batch.items()}
                    model_rep = model.encode_motion(batch)
                    model_reps.append(model_rep)
                    text_rep = model.encode_text(batch['clip_text'])
                    text_reps.append(text_rep)

            model_reps = torch.cat(model_reps, dim=0)
            text_reps = torch.cat(text_reps, dim=0)
            model_reps.requires_grad_()
            
            loss = clip_loss(model_reps, text_reps)
             
            wandb.log({f'{mode}_loss': loss})
            loss.backward()

            # Create cache
            encoder_cache = model_reps.grad
            loss = loss.detach()
        
        
        counter = 0
        start_idx = 0
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            # Put everything in device
            # Added if is_tensor as 'clip_text' in batch is a list of strings, not a tensor!
            batch = {key: val.to(device) if torch.is_tensor(val) else val for key, val in batch.items()}
            counter += 1
            
            #fwd pass
            loss, losses = model(batch)
            if use_cache and mode == "train":
                encoder_gradient = encoder_cache[start_idx:start_idx + batch['z'].shape[0]]
                batch['motion_features'].backward(encoder_gradient, retain_graph=True)
                continue
            
            
            if i == 0:
                dict_loss = deepcopy(losses)
            else:
                for key in dict_loss.keys():
                    if (i + 1) % 51 == 0:
                        wandb.log({key: losses[key]})
                    dict_loss[key] += losses[key]

            if mode == "train":
                loss.barckward()
                optimizer.step()
                optimizer.zero_grad()
        
        if use_cache:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= counter
            optimizer.step()
        
    torch.cuda.empty_cache()
    return dict_loss


def train(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="train")


def test(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="test")
