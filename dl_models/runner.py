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

from common_filepaths import DATA_FOLDER, SUBSET_DATA_FOLDER

PRETRAINED_MODEL_PATH = '/data/therealgabeguo/embedding_net_weights_printsgan.pth'
POSTRAINED_MODEL_PATH = '/data/therealgabeguo/embedding_net_weights.pth'

batch_size=64
test_batch_size=16

training_dataset = TripletDataset(FingerprintDataset(os.path.join(SUBSET_DATA_FOLDER, 'train'), train=True))
#training_dataset = torch.utils.data.Subset(training_dataset, list(range(0, len(training_dataset), 50)))
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TripletDataset(FingerprintDataset(os.path.join(SUBSET_DATA_FOLDER, 'val'), train=False))
#val_dataset = torch.utils.data.Subset(val_dataset, list(range(0, len(val_dataset), 5)))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TripletDataset(FingerprintDataset(os.path.join(SUBSET_DATA_FOLDER, 'test'), train=False))
#test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, len(test_dataset), 5)))
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

# SHOW IMAGES
"""
import matplotlib.pyplot as plt
for the_name, the_dataloader in zip(['train', 'val', 'test'], [train_dataloader, val_dataloader, test_dataloader]):
    print(the_name)
    it = iter(the_dataloader)
    for i in range(10):
        images, labels, filepaths = next(it)
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

# CREATE EMBEDDER
pretrained=False # on image net
embedder = EmbeddingNet(pretrained=pretrained)
log += 'pretrained on image net: {}\n'.format(pretrained)
print('pretrained on image net:', pretrained)

# load saved weights!
pretrained_other_data = True
if pretrained_other_data:
    embedder.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))

pretrained_other_msg = 'pretrained on other data: {}\n'.format(pretrained_other_data)
print(pretrained_other_msg)
log += pretrained_other_msg

"""
# freeze all layers except the last one
n_layers = list(embedder.feature_extractor.children())
print(n_layers)
print(len(n_layers))
n_frozen_layers = 4
for the_param in list(embedder.feature_extractor.children())[:n_frozen_layers]:
    log += 'freezing{}\n'.format(the_param)
    print('freezing {}'.format(the_param))
    the_param.requires_grad = False
"""

# CREATE TRIPLET NET
triplet_net = TripletNet(embedder)

# TRAIN

learning_rate = 0.001
scheduler=None # not needed for Adam
optimizer = optim.Adam(triplet_net.parameters(), lr=learning_rate)
tripletLoss_margin = 1

log += 'learning rate = {}\ntriplet loss margin = {}\n'.format(learning_rate, tripletLoss_margin)

best_val_epoch, best_val_loss = 0, 0

best_val_epoch, best_val_loss = fit(train_loader=train_dataloader, val_loader=val_dataloader, model=triplet_net, \
    loss_fn=nn.TripletMarginLoss(margin=tripletLoss_margin), optimizer=optimizer, scheduler=scheduler, \
    n_epochs=50, cuda='cuda:1', log_interval=100, metrics=[], start_epoch=0, early_stopping_interval=25)

log += 'best_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss)
print('best_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss))

# distances between embedding of positive and negative pair
_01_dist = []
_02_dist = []
dist = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

# SAVE MODEL
torch.save(embedder.state_dict(), POSTRAINED_MODEL_PATH)

# LOAD MODEL
embedder.load_state_dict(torch.load(POSTRAINED_MODEL_PATH))
embedder.eval()
embedder = embedder.to('cuda:1')

# TEST

for i in range(len(test_dataloader)):
    test_images, test_labels, test_filepaths = next(iter(test_dataloader))

    test_images = [item.to('cuda:1') for item in test_images]

    embeddings = [torch.reshape(e, (test_batch_size, e.size()[1])) for e in triplet_net(*test_images)]

    for batch_index in range(test_batch_size):
        _01_dist.append(dist(embeddings[0][batch_index], embeddings[1][batch_index]).item())
        _02_dist.append(dist(embeddings[0][batch_index], embeddings[2][batch_index]).item())
        if math.isnan(_01_dist[-1]):
            print('nan: {}, {}'.format(embeddings[0][batch_index], embeddings[1][batch_index]))
        if math.isnan(_02_dist[-1]):
            print('nan: {}, {}'.format(embeddings[0][batch_index], embeddings[2][batch_index]))

    if i % 200 == 0:
        print('Batch {} out of {}'.format(i, len(test_dataloader)))
        print('\taverage cosine sim between matching pairs:', np.mean(np.array(_01_dist)))
        print('\taverage cosine sim between non-matching pairs:', np.mean(np.array(_02_dist)))
    
    #print(test_filepaths, test_labels)

_01_dist = np.array(_01_dist)
_02_dist = np.array(_02_dist)

log += 'average cosine sim b/w matching pairs: {}\n'.format(np.mean(_01_dist))
log += 'average cosine sim b/w nonmatching pairs: {}\n'.format(np.mean(_02_dist))

print('number of testing positive pairs:', len(_01_dist))
print('number of testing negative pairs:', len(_02_dist))

print('average cosine sim between matching pairs:', np.mean(_01_dist))
print('average cosine sim between non-matching pairs:', np.mean(_02_dist))

from datetime import datetime
datetime_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
with open('/data/therealgabeguo/results/results_{}.txt'.format(datetime_str), 'w') as fout:
    fout.write(log + '\n')
torch.save(embedder.state_dict(), '/data/therealgabeguo/results/weights_{}.pth'.format(datetime_str))

