import os
import sys
sys.path.append('.')

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from src.train.trainer import train
from src.utils.tensors import collate
import src.utils.fixseed  # noqa
from src.utils.action_classifier import evaluate
from src.parser.training import parser
from src.utils.get_model_and_data import get_model_and_data

def do_epochs(model, datasets, parameters, optimizer, writer):
    dataset = datasets["train"]
    test_dataset = datasets["test"]
    print(dataset.__getitem__(0)['clip_text'])
    train_iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
                                shuffle=True, num_workers=8, collate_fn=collate)
    test_iterator = DataLoader(test_dataset, batch_size=160,
                      shuffle=False, num_workers=16, collate_fn=collate)

    logpath = os.path.join(parameters["folder"], "training.log")
    
    with open(logpath, "w") as logfile:
        for epoch in range(1, parameters["num_epochs"]+1):
            wandb.log({'epoch': epoch})
            dict_loss = train(model, optimizer, train_iterator, model.device)
            for key in dict_loss.keys():
                dict_loss[key] /= len(train_iterator)
                writer.add_scalar(f"Loss/{key}", dict_loss[key], epoch)

            epochlog = f"Epoch {epoch}, train losses: {dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)
            writer.flush()
            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
                checkpoint_path = os.path.join(parameters["folder"],
                                               'checkpoint_{:04d}.pth.tar'.format(epoch))
                print('Saving checkpoint {}'.format(checkpoint_path))
                model.eval()
                top_1, top_5 = evaluate(model, test_dataset, test_iterator, parameters)
                model.train()
                wandb.log({'top_1_acc': top_1})
                wandb.log({'top_5_acc': top_5})
                if parameters.get('clip_training', '') == '':
                    state_dict_wo_clip = {k: v for k,v in model.state_dict().items() if not k.startswith('clip_model.')}
                else:
                    state_dict_wo_clip = model.state_dict()
                torch.save(state_dict_wo_clip, checkpoint_path)
                

            writer.flush()


if __name__ == '__main__':
    # parse options
    parameters = parser()
    # logging tensorboard
    writer = SummaryWriter(log_dir=parameters["folder"])
    parameters['only_60_classes'] = True
    parameters['clip_training'] = True
    model, datasets = get_model_and_data(parameters, split="all")
    # parameters['use_action_cat_as_text_labels'] = False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"], weight_decay=0.2)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training model..")
    wandb.login(key='93443c480bfbaa0b19be76d24f2efeb6be3319fd')
    wandb.init(project='text2motion', name='CLIP_SIMPLE')
    print(parameters)
    do_epochs(model, datasets, parameters, optimizer, writer)

    writer.close()
