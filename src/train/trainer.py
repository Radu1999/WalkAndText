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
        if use_cache:
            with torch.no_grad():
                for i, batch in tqdm(enumerate(iterator), desc="Computing batch for caching"):
                    # Put everything in device
                    # Added if is_tensor as 'clip_text' in batch is a list of strings, not a tensor!
                    batch = {key: val.to(device) if torch.is_tensor(val) else val for key, val in batch.items()}
                    # forward pass
                    model_rep = model.encode_motion(batch)
                    model_reps.append(model_rep)
                    text_rep = model.encode_text(batch['clip_text'])
                    text_reps.append(text_rep)
                    # image_rep = model.encode_image(batch['clip_images'], method="clip")
                    # image_reps.append(image_rep)

            model_reps = torch.cat(model_reps, dim=0)
            text_reps = torch.cat(text_reps, dim=0)
            model_reps.requires_grad_()
            
            loss = model.loss(text_reps, model_reps)
            dict_loss = {'loss': loss.item()}
            wandb.log(dict_loss)

            loss.backward()

            # Create cache
            reps_grad_cache = model_reps.grad
            loss = loss.detach()
        
        
        counter = 0
        start_idx = 0
        for i, batch in tqdm(enumerate(iterator), desc=f"Computing batch for {mode}"):
            # Put everything in device
            # Added if is_tensor as 'clip_text' in batch is a list of strings, not a tensor!
            batch = {key: val.to(device) if torch.is_tensor(val) else val for key, val in batch.items()}
            counter += 1

            if use_cache:
                model_rep = model.encode_motion(batch)
            else:
                # update the gradients to zero
                optimizer.zero_grad()

                # forward pass
                loss, losses = model(batch)
            
            if not use_cache:
                if i == 0:
                    dict_loss = deepcopy(losses)
                else:
                    for key in dict_loss.keys():
                        if (i + 1) % 51 == 0:
                            wandb.log({key: losses[key]})
                        dict_loss[key] += losses[key]

            if mode == "train":
                # backward pass
                if use_cache:
                    reps_grad = reps_grad_cache[start_idx:start_idx + model_rep.shape[0]]
                    if reps_grad.shape[0] < model_rep.shape[0]:
                        padding_size = model_rep.shape[0] - reps_grad.shape[0]
                        # Pad reps_grad along the first dimension (0th dimension) with zeros
                        reps_grad = F.pad(reps_grad, (0, 0, 0, padding_size), mode='constant', value=0)
                    start_idx += model_rep.shape[0]
                    model_rep.backward(reps_grad)
                else:
                    loss.backward()
                    optimizer.step()
                
        
        if use_cache and mode == "train":
            # for param in model.parameters():
            #     if param.grad is not None:
            #         param.grad /= counter
            optimizer.step()
            torch.cuda.empty_cache()
            
    return dict_loss


def train(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="train")


def test(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="test")
