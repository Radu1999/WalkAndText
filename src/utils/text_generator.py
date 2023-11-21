import json
import time
import torch
import joblib
from src.datasets.amass import AMASS
import clip

def create_dataset(path: str):
    splits = ['train', 'vald']
    clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda',
                                            jit=False)  # Must set jit=False for training
    data = AMASS(datapath='data/amass_db/babel_30fps_db.pt', clip_preprocess=clip_preprocess,
                split=split,
                num_frames=60)
    total_examples += len(data)
    with open(f'experiments/{path}/clip_texts_train_exp1.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for (i, row) in enumerate(spamreader):
            data.set_clip_text(i, row['Generated caption'])

    joblib.dump(data, f'experiments/{path}/dataset_train.pt')
    return total_examples

if __name__ == '__main__':
    count = create_dataset('exp1')
    print(f'Data generated! For {count} examples it took {time.time() - start} seconds')
