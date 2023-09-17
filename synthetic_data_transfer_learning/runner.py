import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import os
import math

sys.path.append('../')

from trainer import *
from siamese_datasets import *
from fingerprint_dataset import *
from embedding_models import *

from common_filepaths import SYNTHETIC_DATA_FOLDER

import argparse

import wandb

print('synthetic data pretraining')

def main():
    parser = argparse.ArgumentParser(description='Pretraining on PrinstGAN')

    parser.add_argument('--model_path', type=str, default='/data/therealgabeguo/embedding_net_weights_printsgan.pth',
                        help='Where to save the model at the end')
    parser.add_argument('--data_path', type=str, default=SYNTHETIC_DATA_FOLDER,
                        help='Where the PrintsGAN data is')
    parser.add_argument('--output_folder', type=str, default='/data/therealgabeguo/pretrain_results',
                        help='Path to the output folder')
    parser.add_argument('--wandb_project', type=str, default='fingerprint_correlation', \
                        help='database name for wandb')

    args = parser.parse_args()

    if args.wandb_project:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=args.wandb_project,
            name=args.model_path.split('/')[-1],
            # Track hyperparameters and run metadata
            config=vars(args)
        )

    # You can now access the values as:
    # args.model_path and args.output_folder

    print(f'Model path: {args.model_path}')
    print(f'Output folder: {args.output_folder}')

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    batch_size=64

    training_dataset = TripletDataset(FingerprintDataset(os.path.join(args.data_path, 'train'), train=True))
    #training_dataset = torch.utils.data.Subset(training_dataset, list(range(0, len(training_dataset), 50)))
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    val_dataset = TripletDataset(FingerprintDataset(os.path.join(args.data_path, 'val'), train=False))
    #val_dataset = torch.utils.data.Subset(val_dataset, list(range(0, len(val_dataset), 50)))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    # CLEAR CUDA MEMORY
    # https://stackoverflow.com/questions/54374935/how-to-fix-this-strange-error-runtimeerror-cuda-error-out-of-memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # FILE OUTPUT
    log = ""

    # CREATE EMBEDDER
    pretrained=False
    embedder = EmbeddingNet(pretrained=pretrained)
    log += 'pretrained: {}\n'.format(pretrained)
    print('pretrained:', pretrained)

    # CREATE TRIPLET NET
    triplet_net = TripletNet(embedder)

    if args.wandb_project:
        wandb.summary['model'] = str(triplet_net)
        wandb.watch(triplet_net, log='all', log_freq=500)

    # TRAIN

    learning_rate = 0.001
    n_epochs = 25
    optimizer = optim.Adam(triplet_net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, 
                                                     eta_min=learning_rate*1e-3, last_epoch=- 1, verbose=False)
    tripletLoss_margin = 0.2

    log += 'learning rate = {}\ntriplet loss margin = {}\n'.format(learning_rate, tripletLoss_margin)

    best_val_epoch, best_val_loss = 0, 0

    best_val_epoch, best_val_loss, all_epochs, train_losses, val_losses = fit(
        train_loader=train_dataloader, val_loader=val_dataloader, model=triplet_net, \
        loss_fn=nn.TripletMarginLoss(margin=tripletLoss_margin), optimizer=optimizer, scheduler=scheduler, \
        n_epochs=n_epochs, cuda='cuda', log_interval=1000, metrics=[], start_epoch=0, early_stopping_interval=5
    )

    log += 'best_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss)
    log += f'\nall epochs: {all_epochs}\ntrain losses: {train_losses}\nval losses: {val_losses}\n'
    print(log)

    # SAVE MODEL
    torch.save(embedder.state_dict(), args.model_path)

    from datetime import datetime
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(os.path.join(args.output_folder, 'results_{}.txt'.format(datetime_str)), 'w') as fout:
        fout.write(log + '\n')
    torch.save(embedder.state_dict(), os.path.join(args.output_folder, 'weights_{}.pth'.format(datetime_str)))

if __name__ == "__main__":
    main()
