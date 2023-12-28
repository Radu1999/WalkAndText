import os
import sys
sys.path.append('.')
os.environ['TOKENIZERS_PARALLELISM'] = 'True'

import torch
import math
import wandb
import joblib
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader
from src.train.trainer import train, test
from src.utils.tensors import collate
import src.utils.fixseed  # noqa
from src.utils.action_classifier import evaluate, evaluate_transformer_classifier
from src.parser.training import parser
from src.utils.get_model_and_data import get_model_and_data
from src.utils.misc import load_model_wo_clip

def do_epochs(model, datasets, parameters, optimizer, writer, scheduler):
    dataset = datasets["train"]
    #dataset.sampling = 'random_conseq'
    test_dataset = datasets["test"]
    #print(dataset.__getitem__(0)['clip_text'])
    test_iterator = DataLoader(test_dataset, batch_size=160,
                      shuffle=False, num_workers=16, collate_fn=collate)

    logpath = os.path.join(parameters["folder"], "training.log")
    batch_size = 80
    interval = 2
    counter = 0
    train_iterator = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=16, collate_fn=collate)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=2000, eta_min=0)
    with open(logpath, "w") as logfile:
        for epoch in range(1, parameters["num_epochs"]+1):
            counter += 1
            wandb.log({'epoch': epoch})
            wandb.log({'batch size': batch_size})
            dict_loss = train(model, optimizer, scheduler, train_iterator, model.device )
            if scheduler is not None:
                scheduler.step()
            for key in dict_loss.keys():
                dict_loss[key] /= len(train_iterator)
                writer.add_scalar(f"Loss/{key}", dict_loss[key], epoch)

            epochlog = f"Epoch {epoch}, train losses: {dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)
            writer.flush()
            dict_loss = test(model, optimizer, test_iterator, model.device)
            
            for key in dict_loss.keys():
                dict_loss[key] /= len(test_iterator)
                wandb.log({f'val_{key}': dict_loss[key]})
                writer.add_scalar(f"Loss/{key}", dict_loss[key], epoch)

            epochlog = f"Epoch {epoch}, val losses: {dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)
            writer.flush()
                
            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
                
                model.eval()
                if parameters.get("model", "default") != "default":
                    top_1, top_5 = evaluate_transformer_classifier(model, test_dataset, test_iterator, parameters)
                else:
                    top_1, top_5 = evaluate(model, test_dataset, test_iterator, parameters)
                model.train()
                wandb.log({'top_1_acc': top_1})
                wandb.log({'top_5_acc': top_5})
                print(f'Top 1: {top_1}')
                print(f'Top 5: {top_5}')
                checkpoint_path = os.path.join(parameters["folder"],
                                               'checkpoint_{:04d}.pth.tar'.format(epoch))
                if top_1 > 0.46:
                    print('Saving checkpoint {}'.format(checkpoint_path))
                    torch.save(model.state_dict(), checkpoint_path)
                
            # if counter % interval == 0 and batch_size < 128:
            #     batch_size *= 2
            #     interval *= 2
            #     counter = 0
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = param_group['lr'] * math.sqrt(2)
            #     train_iterator = DataLoader(dataset, batch_size=batch_size,
            #                 shuffle=True, num_workers=16, collate_fn=collate)
            writer.flush()


if __name__ == '__main__':
    # parse options
    parameters = parser()
    # logging tensorboard
    writer = SummaryWriter(log_dir=parameters["folder"])
    parameters['only_60_classes'] = True
    parameters['clip_training'] = True
    # parameters['sampling'] = 'random'
    #parameters['model'] = 'classifier'
    #text_descriptions = joblib.load('./data/multiple_captions.pt')
    model, datasets = get_model_and_data(parameters, split="all", descriptions=None)
    
    # checkpointpath = os.path.join('./exps/pretraining', 'checkpoint_0020.pth.tar')
    # state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    # load_model_wo_clip(model, state_dict)
    
    # parameters['use_action_cat_as_text_labels'] = False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"], weight_decay=0.1)
    scheduler = None # CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training model..")
    wandb.login(key='93443c480bfbaa0b19be76d24f2efeb6be3319fd')
    wandb.init(project='text2motion', name='CLIP_SIMPLE')
    print(parameters)
    do_epochs(model, datasets, parameters, optimizer, writer, scheduler)

    writer.close()
