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
    dict_loss = {}
    model_reps = []
    text_reps = []
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    encoder_cache = []
    with grad_env():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            if len(batch['y']) == 0 or batch['x'].shape[0] == 1:
                continue
            batch = {key: val.to(device) if torch.is_tensor(val) and key != 'y'  else val for key, val in batch.items()}

            loss, losses = model(batch)
            if loss == 0:
                print(batch['motion_features'].size())
            if i == 0:
                dict_loss = deepcopy(losses)
            else:
                for key in dict_loss.keys():
                    if (i + 1) % 51 == 0:
                        wandb.log({key: losses[key]})
                        
                    dict_loss[key] += losses[key]
                
            if mode == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
    torch.cuda.empty_cache()
    return dict_loss


def train(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="train")


def test(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="test")
