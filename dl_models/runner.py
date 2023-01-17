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
from losses import *
from siamese_datasets import *
from fingerprint_dataset import *
from embedding_models import *

from common_filepaths import *

# ResNet-18
PRETRAINED_MODEL_PATH = '/data/therealgabeguo/embedding_net_weights_printsgan.pth'
POSTRAINED_MODEL_PATH = '/data/therealgabeguo/embedding_net_weights.pth'

batch_size=64 # ResNet-18 & 34
num_accumulated_batches=1 # ResNet-18 & 34

the_data_folders = [DATA_FOLDER, EXTRA_DATA_FOLDER, UB_DATA_FOLDER]
val_data_folders = [DATA_FOLDER, EXTRA_DATA_FOLDER, UB_DATA_FOLDER]

train_dir_paths = [os.path.join(x, 'train') for x in the_data_folders]
val_dir_paths = [os.path.join(x, 'val') for x in val_data_folders]

training_dataset = TripletDataset(FingerprintDataset(train_dir_paths, train=True))
#training_dataset = torch.utils.data.Subset(training_dataset, list(range(0, len(training_dataset), 50)))
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

val_dataset = TripletDataset(FingerprintDataset(val_dir_paths, train=False))
#val_dataset = torch.utils.data.Subset(val_dataset, list(range(0, len(val_dataset), 5)))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

# SHOW IMAGES
"""
import matplotlib.pyplot as plt
for the_name, the_dataloader in zip(['train', 'val', 'test'], [train_dataloader, val_dataloader, test_dataloader]):
    print(the_name)
    it = iter(the_dataloader)
    for i in range(10):
        images, labels, filepaths = next(it)
        #print(images[0].shape)
        #print('mean:', [torch.mean(images[0][:,i]) for i in range(3)])
        #print('std:', [torch.std(images[0][:,i]) for i in range(3)])
        next_img = images[2][0]
        the_min = torch.min(next_img)
        the_max = torch.max(next_img)
        next_img = (next_img - the_min) / (the_max - the_min)
        print(filepaths[2][0])
        print(next_img[0])
        plt.imshow(next_img.permute(1, 2, 0))
        plt.show()
"""

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
pretrained=False # on image net
embedder = EmbeddingNet(pretrained=pretrained)
log += 'pretrained on image net: {}\n'.format(pretrained)
print('pretrained on image net:', pretrained)

# load saved weights!
pretrained_other_data = True # weights from PrintsGAN (or SD300)
if pretrained_other_data:
    embedder.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))

pretrained_other_msg = 'pretrained on other data: {}, {}\n'.format(pretrained_other_data, PRETRAINED_MODEL_PATH)
print(pretrained_other_msg)
log += pretrained_other_msg

# CREATE TRIPLET NET
triplet_net = TripletNet(embedder)

# TRAIN

learning_rate = 0.001 # ResNet-18 & 34
scheduler=None # not needed for Adam
optimizer = optim.Adam(triplet_net.parameters(), lr=learning_rate)
tripletLoss_margin = 0.2

log += 'learning rate = {}\ntriplet loss margin = {}\n'.format(learning_rate, tripletLoss_margin)

best_val_epoch, best_val_loss = 0, 0

best_val_epoch, best_val_loss = fit(train_loader=train_dataloader, val_loader=val_dataloader, model=triplet_net, \
    loss_fn=nn.TripletMarginLoss(margin=tripletLoss_margin), optimizer=optimizer, scheduler=scheduler, \
    n_epochs=100, cuda='cuda:1', log_interval=300, metrics=[], start_epoch=0, early_stopping_interval=65, \
    num_accumulated_batches=num_accumulated_batches)

log += 'best_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss)
print('best_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss))

# SAVE MODEL
torch.save(embedder.state_dict(), POSTRAINED_MODEL_PATH)

from datetime import datetime
datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
with open('/data/therealgabeguo/results/results_{}.txt'.format(datetime_str), 'w') as fout:
    fout.write(log + '\n')
torch.save(embedder.state_dict(), '/data/therealgabeguo/results/weights_{}.pth'.format(datetime_str))

