"""
Version of runner for investigating inherent feature correlations, e.g., ridges, minutiae.
By default, no pretraining, since that would introduce bias.
Input dataset representing desired features. Also input CUDA you want.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import os
import math
import getopt, argparse

sys.path.append('../')

from trainer import *
from siamese_datasets import *
from fingerprint_dataset import *
from embedding_models import *

from common_filepaths import *

def main(datasets, the_cuda, \
        batch_size=64, num_accumulated_batches=1, \
        pretrained_image_net=False, pretrained_printsgan=False, \
        PRETRAINED_MODEL_PATH='/data/therealgabeguo/embedding_net_weights_printsgan.pth', \
        learning_rate=0.001, scheduler=None, tripletLoss_margin=0.2, \
        num_epochs=200, early_stopping_interval=65):
    
    the_name = '_'.join([path[:len(path) if path[-1] != '/' else -1].split('/')[-1] for path in datasets])
    # ResNet-18
    POSTRAINED_MODEL_PATH = '/data/therealgabeguo/fingerprint_weights/embedding_net_weights_' + the_name + '.pth'
    print('weights saved to:', POSTRAINED_MODEL_PATH)

    train_dir_paths = [os.path.join(x, 'train') for x in datasets]
    val_dir_paths = [os.path.join(x, 'val') for x in datasets]

    training_dataset = TripletDataset(FingerprintDataset(train_dir_paths, train=True))
    #training_dataset = torch.utils.data.Subset(training_dataset, list(range(0, len(training_dataset), 50)))
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    val_dataset = TripletDataset(FingerprintDataset(val_dir_paths, train=False))
    #val_dataset = torch.utils.data.Subset(val_dataset, list(range(0, len(val_dataset), 5)))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    # CLEAR CUDA MEMORY
    # https://stackoverflow.com/questions/54374935/how-to-fix-this-strange-error-runtimeerror-cuda-error-out-of-memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # FILE OUTPUT
    log = ""

    # LOG TRAINING DATA
    log += 'Training data: {}\n'.format(train_dir_paths)
    print('Training data: {}\n'.format(train_dir_paths))

    # CREATE EMBEDDER
    embedder = EmbeddingNet(pretrained=pretrained_image_net)
    log += 'pretrained on image net: {}\n'.format(pretrained_image_net)
    print('pretrained on image net:', pretrained_image_net)

    # load saved weights!
    if pretrained_printsgan:
        embedder.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))

    pretrained_other_msg = 'pretrained on other data: {}, {}\n'.format(pretrained_printsgan, PRETRAINED_MODEL_PATH)
    print(pretrained_other_msg)
    log += pretrained_other_msg

    # CREATE TRIPLET NET
    triplet_net = TripletNet(embedder)

    # TRAIN
    optimizer = optim.Adam(triplet_net.parameters(), lr=learning_rate)

    log += 'learning rate = {}\ntriplet loss margin = {}\n'.format(learning_rate, tripletLoss_margin)
    print('learning rate = {}\ntriplet loss margin = {}\n'.format(learning_rate, tripletLoss_margin))
    log += 'max epochs = {}\n'.format(num_epochs)
    print('max epochs = {}\n'.format(num_epochs))

    best_val_epoch, best_val_loss = 0, 0

    best_val_epoch, best_val_loss = fit(train_loader=train_dataloader, val_loader=val_dataloader, model=triplet_net, \
        loss_fn=nn.TripletMarginLoss(margin=tripletLoss_margin), optimizer=optimizer, scheduler=scheduler, \
        n_epochs=num_epochs, cuda=the_cuda, log_interval=300, metrics=[], start_epoch=0, early_stopping_interval=early_stopping_interval, \
        num_accumulated_batches=num_accumulated_batches, temp_model_path='temp_{}.pth'.format(the_name))

    log += 'best_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss)
    print('best_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss))

    # SAVE MODEL
    torch.save(embedder.state_dict(), POSTRAINED_MODEL_PATH)
    log += 'save to: {}\n'.format(POSTRAINED_MODEL_PATH)

    from datetime import datetime
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open('/data/therealgabeguo/results/results_{}.txt'.format(datetime_str), 'w') as fout:
        fout.write(log + '\n')
    torch.save(embedder.state_dict(), '/data/therealgabeguo/results/weights_{}.pth'.format(datetime_str))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser('parameterized_runner.py')
    parser.add_argument('--datasets', '-d', help='Paths to folders containing images', type=str)
    parser.add_argument('--cuda', '-c', help='Name of GPU we want to use', type=str)
    args = parser.parse_args()
    datasets = args.datasets.split()
    cuda = args.cuda

    main(datasets, cuda)