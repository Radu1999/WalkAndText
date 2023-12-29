import torch
from tqdm import tqdm
from copy import deepcopy
import pickle
import clip
import wandb
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from itertools import islice

def train_or_test(model, optimizer, scheduler, iterator, device, mode="train"):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")
        


    # loss of the epoch
    dict_loss = {}
    model_reps = []
    text_reps = []
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    use_cache = False
    simulated_bs = 1
    encoder_cache = []
    with grad_env():
        counter = 0
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            # Put everything in device
            # Added if is_tensor as 'clip_text' in batch is a list of strings, not a tensor!
            if len(batch['clip_text']) == 0 or batch['x'].shape[0] == 1:
                continue
            batch = {key: val.to(device) if torch.is_tensor(val) and key != 'mask' and key != 'y'  else val for key, val in batch.items()}
            batch['mask'] = batch['mask'].to(device)
            batch['y'] = batch['y'].to(device)
            counter += 1
            
            #fwd pass
            #with torch.cuda.amp.autocast():
            loss, losses = model(batch)
            if loss == 0:
                print(batch['motion_features'].size())
            if i == 0:
                dict_loss = deepcopy(losses)
            else:
                if (i + 1) % 51 == 0:
                    if scheduler is not None:
                        wandb.log({'lr': scheduler.get_last_lr()[0]})
                for key in dict_loss.keys():
                    if (i + 1) % 51 == 0:
                        wandb.log({key: losses[key]})
                        
                    dict_loss[key] += losses[key]
                    
            if use_cache and mode == "train":
                # build cache in advance
                if (i + 1) % simulated_bs == 0:
                    model_reps = []
                    text_reps = []
                    start_idx = 0
                    future_batches = list(islice(iterator, simulated_bs))
                    for future_batch in future_batches:
                        future_batch = {key: val.to(device) if torch.is_tensor(val) else val for key, val in future_batch.items()}
                        with torch.no_grad():
                            model_rep = model.encode_motion(future_batch)
                            model_reps.append(model_rep)
                            text_rep = model.encode_text(future_batch['clip_text'])
                            text_reps.append(text_rep)
                    model_reps = torch.cat(model_reps, dim=0)
                    text_reps = torch.cat(text_reps, dim=0)
                    model_reps.requires_grad_()

                    cache_loss = model.loss(model_reps, text_reps)
                    cache_loss.backward()

                    # Create cache
                    encoder_cache = model_reps.grad
                        
                
                encoder_gradient = encoder_cache[start_idx:start_idx + batch['motion_features'].shape[0]]
                if encoder_gradient.shape[0] < batch['motion_features'].shape[0]:
                    padding_size = batch['motion_features'].shape[0] - encoder_gradient.shape[0]
                    pad_values = (0, 0, 0, padding_size)
                    encoder_gradient = nn.functional.pad(encoder_gradient, pad_values, value=0)
                loss.backward(encoder_gradient)
                start_idx += batch['motion_features'].shape[0]
                
                if (i + 1) % simulated_bs == 0:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad /= counter
                    optimizer.step()
                    simulated_bs_idx += 1
                    start_idx = 0
                continue
                
            if mode == "train":
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                     scheduler.step()
                optimizer.zero_grad()
        
    torch.cuda.empty_cache()
    return dict_loss


def train(model, optimizer, scheduler, iterator, device):
    return train_or_test(model, optimizer, scheduler, iterator, device, mode="train")


def test(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, None, iterator, device, mode="test")
