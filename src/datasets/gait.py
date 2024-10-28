import os
import numpy as np
import joblib
from torch.utils.data import Dataset
import sys
import torch
import csv
from ..utils.misc import to_torch

sys.path.append('')

class Gait(Dataset):
    def __init__(self, sampling="random", sampling_step=1, datapath="caption_retrieval_dataset_test.pt", num_frames=60, **kwargs):
        self.num_frames = num_frames
        self.datapath = datapath
        self.db = self.load_db()
        self.sampling = sampling
        self.sampling_step = sampling_step
        self.label_mapping = {
          "male": 0,
          "senior": 1,
          "carry something in hand": 2,
          "backpack": 3,
          "boots": 4,
          "sneakers": 5,
          "dress": 6,
          "young-adult": 7,
          "front": 8,
          "adult": 9,
          "gathering": 10,
          "cloth shoes": 11,
          "female": 12,
          "middle-aged adult": 13,
          "overweight": 14,
          "hand trunk": 15,
          "casual": 16,
          "long coat": 17,
          "long trousers": 18,
          "side": 19,
          "talking": 20,
          "slim": 21,
          "normal weight": 22,
          "calling": 23,
          "formal": 24,
          "jacket": 25,
          "leather shoes": 26,
          "trousers": 27,
          "shoes": 28,
          "sandals": 29,
          "sports shoes": 30,
          "bag": 31,
          "child": 32,
          "back": 33,
          "pulling": 34,
          "holding something": 35
        }
    
    def __getitem__(self, index):
        return self._get_item_data_index(index)

    def load_db(self):
        db_file = self.datapath
        db = joblib.load(db_file)
        return db
    
    def get_pose_data(self, data_index, frame_ix):
        joints = self.db[data_index]['joints'][frame_ix]
        joints = joints - joints[0,0,:]
        joints = to_torch(joints)
        joints = joints.permute(1, 2, 0).contiguous()
        joints = joints.float()
        
        caption = self.db[data_index]['text']
        feat_id = self.db[data_index]['feature_id']
        labels = [0] * len(self.label_mapping)
        for feat in feat_id.split(','):
            labels[self.label_mapping[feat]] = 1.0
        return joints, caption, feat_id, labels
        

    def _get_item_data_index(self, data_index):
        nframes = self.db[data_index]['joints'].shape[0]
        num_frames = self.num_frames
        # sampling goal: input: ----------- 11 nframes
        #                       o--o--o--o- 4  ninputs
        #
        # step number is computed like that: [(11-1)/(4-1)] = 3
        #                   [---][---][---][-
        # So step = 3, and we take 0 to step*ninputs+1 with steps
        #                   [o--][o--][o--][o-]
        # then we can randomly shift the vector
        #                   -[o--][o--][o--]o
        # If there are too much frames required
        if num_frames > nframes:
            # adding the last frame until done
            ntoadd = max(0, num_frames - nframes)
            lastframe = nframes - 1
            padding = lastframe * np.ones(ntoadd, dtype=int)
            frame_ix = np.concatenate((np.arange(0, nframes),
                                       padding))

        elif self.sampling in ["conseq", "random_conseq"]:

            step_max = (nframes - 1) // (num_frames - 1)
            if self.sampling == "conseq":
                step = self.sampling_step
            elif self.sampling == "random_conseq":
                step = random.randint(1, step_max)

            lastone = step * (num_frames - 1)
            shift_max = nframes - lastone - 1
            shift = random.randint(0, max(0, shift_max - 1))
            frame_ix = shift + np.arange(0, lastone + 1, step)

        elif self.sampling == "random":
            choices = np.random.choice(range(nframes),
                                       num_frames,
                                       replace=False)
            frame_ix = sorted(choices)

        else:
            raise ValueError("Sampling not recognized.")

        pose, caption, feat_id, labels = self.get_pose_data(data_index, frame_ix)
        output = {'inp': pose, 'target': caption, 'id': feat_id, 'labels': labels}
        return output
    
    def __len__(self):
        return len(self.db)


if __name__ == "__main__":
    device = 'cpu'
    dataset = Gait()
    print(len(dataset))
    print(dataset.__getitem__(0)['inp'].shape)
    print(dataset.__getitem__(0)['target'])
