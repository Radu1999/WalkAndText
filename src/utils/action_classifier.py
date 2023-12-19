import sys
sys.path.append('.')
import os
import torch
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.utils.misc import load_model_wo_clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.tensors import collate
import clip
from src.visualize.visualize import get_gpu_device
from src.utils.action_label_to_idx import action_label_to_idx
from ..datasets.get_dataset import get_datasets
import src.utils.fixseed
import joblib
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import torch.nn.functional as F

def evaluate(model, dataset, iterator, parameters):
    TOP_K_METRIC = 5
    
    if 'use_action_cat_as_text_labels' in parameters and parameters['use_action_cat_as_text_labels']:
        ground_truth_gen = list(action_label_to_idx.keys())
        ground_truth_gen.sort(key=lambda x: action_label_to_idx[x])
    else:
        # ground_truth_gen = list(action_label_to_idx.keys())
        # ground_truth_gen.sort(key=lambda x: action_label_to_idx[x])
        ground_truth = joblib.load('./data/babel_llm_1_smaller/grountruth.pt')
        ground_truth.sort(key=lambda x: action_label_to_idx[x['orig']])
        ground_truth_gen = [gt['generated'] for gt in ground_truth]
    ground_truth_gen = ground_truth_gen[:60]
    
    correct_preds_top_5, correct_preds_top_1 = 0,0
    total_samples = 0
    with torch.no_grad():
        
        text_features = model.encode_text(ground_truth_gen)
        # classes_text_emb_norm =  F.normalize(classes_text_emb, p=2, dim=-1)
        for i, batch in enumerate(iterator):
            if isinstance(batch['x'], list):
                continue
            for key in batch.keys():
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(model.device)
                    
            labels = list(map(lambda x: [action_label_to_idx[cat] for cat in x], batch['all_categories']))
            motion_features = model.encode_motion(batch)
            similarity =  model.loss.logit_scale * motion_features @ text_features.t()
            
            total_samples += motion_features.shape[0]
            for i in range(similarity.shape[0]):
                values, indices = similarity[i].topk(TOP_K_METRIC)

                # TOP-5 CHECK
                if any([gt_cat_idx in indices for gt_cat_idx in labels[i]]):
                    correct_preds_top_5 += 1

                # TOP-1 CHECK
                values = values[:1]
                indices = indices[:1]
                if any([gt_cat_idx in indices for gt_cat_idx in labels[i]]):
                    correct_preds_top_1 += 1

            # print(f"Current Top-5 Acc. : {100 * correct_preds_top_5 / total_samples:.2f}%")
        
        top_5_acc = correct_preds_top_5 / total_samples
        top_1_acc = correct_preds_top_1 / total_samples
    return top_1_acc, top_5_acc

    

if __name__ == '__main__':
    parameters, folder, checkpointname, epoch = parser(checkpoint=True)
    #gpu_device = get_gpu_device()
    parameters["device"] = f"cuda"
    data_split = 'all'  # Hardcoded
    # parameters['use_action_cat_as_text_labels'] = True
    parameters['only_60_classes'] = True

    TOP_K_METRIC = 5

    model, datasets = get_model_and_data(parameters, split="vald")

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)
    model.eval()
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=parameters['device'], jit=False)  # Must set jit=False for trainin)
    dataset = datasets["test"]
    iterator = DataLoader(dataset, batch_size=160,
                      shuffle=False, num_workers=16, collate_fn=collate)
    top_1, top_5 = evaluate(model, dataset, iterator, parameters)
    print(top_1)
    print(top_5)