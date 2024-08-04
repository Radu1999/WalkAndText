import os
import sys
sys.path.append('.')
os.environ['TOKENIZERS_PARALLELISM'] = 'True'

import torch
import math
import wandb
import joblib
from torch.utils.data import DataLoader
from src.train.trainer import train, test
from src.utils.tensors import collate_gait
import src.utils.fixseed  # noqa
from src.utils.action_classifier import evaluate_gait_classifier
from src.utils.get_model_and_data import get_model_and_data_gait

def do_epochs(model, datasets, parameters, optimizerwrit):
    dataset = datasets["train"]
    test_dataset = datasets["test"]
    test_iterator = DataLoader(test_dataset, batch_size=8192,
                      shuffle=False, num_workers=16, collate_fn=collate_gait)
    batch_size = parameters["batch_size"]
    
    train_iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
                      shuffle=True, num_workers=16, collate_fn=collate_gait)

    logpath = os.path.join(parameters["folder"], "training.log")
    
    with open(logpath, "w") as logfile:
        for epoch in range(1, parameters["num_epochs"]+1):
            wandb.log({'epoch': epoch})
            wandb.log({'batch size': batch_size})
            dict_loss = train(model, optimizer, train_iterator, model.device )
            for key in dict_loss.keys():
                dict_loss[key] /= len(train_iterator)

            epochlog = f"Epoch {epoch}, train losses: {dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)
            dict_loss = test(model, optimizer, test_iterator, model.device)
            results = evaluate_gait_classifier(model, test_dataset, test_iterator, parameters)
            print(results)
            for (result, key) in zip(results, test_dataset.label_mapping.keys()):
                wandb.log({key: result})
                
            
            for key in dict_loss.keys():
                dict_loss[key] /= len(test_iterator)
                wandb.log({f'val_{key}': dict_loss[key]})

            epochlog = f"Epoch {epoch}, val losses: {dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)
            
            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
                checkpoint_path = os.path.join(parameters["folder"],
                                               'checkpoint_{:04d}.pth.tar'.format(epoch))

                print('Saving checkpoint {}'.format(checkpoint_path))
                torch.save(model.state_dict(), checkpoint_path)
                

if __name__ == '__main__':
    parameters = {
        'sampling': 'random',
        'num_epochs': 200,
        'snapshot': 20,
        'device': 0,
        'folder': './exps',
        'lr': 10e-5,
        'num_layers': 12,
        'num_frames': 60,
        'batch_size': 256,
        'pose_rep': 'rot6d',
        'jointstype': 'vertices',
        'glob_rot': [3.141592653589793, 0, 0],
        'glob': True,
        'latent_dim': 768,
        'translation': True,
        'vertstrans': False,
        
        
    }
    model, datasets = get_model_and_data_gait(parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"], weight_decay=0.1)
    
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training model..")
    
    wandb.login(key='93443c480bfbaa0b19be76d24f2efeb6be3319fd')
    wandb.init(project='text2motion', name='CLIP_SIMPLE')
    do_epochs(model, datasets, parameters, optimizer)
