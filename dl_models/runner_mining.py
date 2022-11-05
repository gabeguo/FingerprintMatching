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
from utils import *
from metrics import *

from common_filepaths import DATA_FOLDER

MODEL_PATH = 'embedding_net_weights.pth'

batch_size=64

training_dataset = FingerprintDataset(os.path.join(DATA_FOLDER, 'train'), train=True)
train_batch_sampler = BalancedBatchSampler(training_dataset.train_labels, n_classes=10, n_samples=25)
online_train_loader = DataLoader(training_dataset, batch_sampler=train_batch_sampler)

val_dataset = FingerprintDataset(os.path.join(DATA_FOLDER, 'val'), train=False)
val_batch_sampler = BalancedBatchSampler(val_dataset.test_labels, n_classes=10, n_samples=25)
val_dataloader = DataLoader(val_dataset, batch_sampler=val_batch_sampler)

test_dataset = FingerprintDataset(os.path.join(DATA_FOLDER, 'test'), train=False)
test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=10, n_samples=25)
test_dataloader = DataLoader(test_dataset, batch_sampler=test_batch_sampler)

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

# FILE OUTPUT
log = ""

# CREATE EMBEDDER

embedder = EmbeddingNet()

# load saved weights!
# embedder.load_state_dict(torch.load(MODEL_PATH))

# CREATE TRIPLET NET
triplet_net = embedder

# TRAIN

learning_rate = 0.005
momentum = 0.99
weight_decay = 5e-5
lr_decay_step=3
lr_decay_factor=0.9
optimizer = optim.Adam(triplet_net.parameters(), lr=learning_rate) #optim.SGD(triplet_net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)
tripletLoss_margin = 1

log += 'learning_rate = {}\nmomentum = {}\nweight_decay = {}\nlr_decay_step = {}\nlr_decay_factor = {}\n'.format(learning_rate, \
        momentum, weight_decay, lr_decay_step, lr_decay_factor)

#
# nn.TripletMarginLoss(margin=tripletLoss_margin)
#

fit(train_loader=online_train_loader, val_loader=val_dataloader, model=triplet_net, \
    loss_fn=OnlineTripletLoss(tripletLoss_margin, SemihardNegativeTripletSelector(tripletLoss_margin)), optimizer=optimizer, scheduler=scheduler, \
    n_epochs=100, cuda='cuda:0', log_interval=10, metrics=[AverageNonzeroTripletsMetric()], start_epoch=0)

# distances between embedding of positive and negative pair
_01_dist = []
_02_dist = []
dist = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

# SAVE MODEL
torch.save(embedder.state_dict(), MODEL_PATH)

# LOAD MODEL
embedder.load_state_dict(torch.load(MODEL_PATH))
embedder.eval()
embedder = embedder.to('cuda:0')

# TEST

for i in range(len(test_dataloader)):
    test_images, test_labels, test_filepaths = next(iter(test_dataloader))

    test_images = [item.to('cuda:0') for item in test_images]

    embeddings = [torch.reshape(e, (batch_size, e.size()[1])) for e in triplet_net(*test_images)]
    # embeddings.shape[0] is (anchor, pos, neg); embeddings.shape[1] is batch size; embeddings.shape[2] is embedding length
    #print([embedding.size() for embedding in embeddings])

    for batch_index in range(batch_size):
        #print(dist(embeddings[0][batch_index], embeddings[1][batch_index]))
        #print(dist(embeddings[0][batch_index], embeddings[2][batch_index]))
        _01_dist.append(dist(embeddings[0][batch_index], embeddings[1][batch_index]).item())
        _02_dist.append(dist(embeddings[0][batch_index], embeddings[2][batch_index]).item())
        if math.isnan(_01_dist[-1]):
            print('nan: {}, {}'.format(embeddings[0][batch_index], embeddings[1][batch_index]))
        if math.isnan(_02_dist[-1]):
            print('nan: {}, {}'.format(embeddings[0][batch_index], embeddings[2][batch_index]))

    if i % 50 == 0:
        print('Batch {} out of {}'.format(i, len(test_dataloader)))
        print('\taverage cosine sim between matching pairs:', np.mean(np.array(_01_dist)))
        print('\taverage cosine sim between non-matching pairs:', np.mean(np.array(_02_dist)))
    
    #print(test_filepaths, test_labels)

_01_dist = np.array(_01_dist)
_02_dist = np.array(_02_dist)

#print(_01_dist[0].size())
#print(_02_dist[0].size())

log += 'average cosine sim b/w matching pairs: {}\n'.format(np.mean(_01_dist))
log += 'average cosine sim b/w nonmatching pairs: {}\n'.format(np.mean(_02_dist))

print('number of testing positive pairs:', len(_01_dist))
print('number of testing negative pairs:', len(_02_dist))

print('average cosine sim between matching pairs:', np.mean(_01_dist))
print('average cosine sim between non-matching pairs:', np.mean(_02_dist))

with open('results_freeze_{}.txt'.format(n_frozen_layers), 'w') as fout:
    fout.write(log + '\n')

