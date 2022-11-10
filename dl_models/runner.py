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

from common_filepaths import DATA_FOLDER

MODEL_PATH = '/data/therealgabeguo/embedding_net_weights.pth'

batch_size=32

training_dataset = TripletDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'train'), train=True))
#training_dataset = torch.utils.data.Subset(training_dataset, list(range(0, len(training_dataset), 50)))
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TripletDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'val'), train=False))
#val_dataset = torch.utils.data.Subset(val_dataset, list(range(0, len(val_dataset), 5)))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TripletDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'test'), train=False))
#test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, len(test_dataset), 5)))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
pretrained=True
embedder = EmbeddingNet(pretrained=pretrained)
log += 'pretrained: {}\n'.format(pretrained)
print('pretrained:', pretrained)

# load saved weights!
#embedder.load_state_dict(torch.load(MODEL_PATH))

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

learning_rate = 0.01
momentum = 0.99
weight_decay = 1e-4
lr_decay_step=2
lr_decay_factor=0.7
optimizer = optim.Adam(triplet_net.parameters(), lr=learning_rate) #optim.SGD(triplet_net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)
tripletLoss_margin = 1

log += 'learning_rate = {}\nmomentum = {}\nweight_decay = {}\nlr_decay_step = {}\nlr_decay_factor = {}\n'.format(learning_rate, \
        momentum, weight_decay, lr_decay_step, lr_decay_factor)

best_val_epoch, best_val_loss = 0, 0

best_val_epoch, best_val_loss = fit(train_loader=train_dataloader, val_loader=val_dataloader, model=triplet_net, \
    loss_fn=nn.TripletMarginLoss(margin=tripletLoss_margin), optimizer=optimizer, scheduler=scheduler, \
    n_epochs=100, cuda='cuda:1', log_interval=100, metrics=[], start_epoch=0, early_stopping_interval=50)


log += 'best_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss)
print('best_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss))

# distances between embedding of positive and negative pair
_01_dist = []
_02_dist = []
dist = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

# SAVE MODEL
torch.save(embedder.state_dict(), MODEL_PATH)

# LOAD MODEL
embedder.load_state_dict(torch.load(MODEL_PATH))
embedder.eval()
embedder = embedder.to('cuda:1')

# TEST

for i in range(len(test_dataloader)):
    test_images, test_labels, test_filepaths = next(iter(test_dataloader))

    test_images = [item.to('cuda:1') for item in test_images]

    embeddings = [torch.reshape(e, (batch_size, e.size()[1])) for e in triplet_net(*test_images)]

    for batch_index in range(batch_size):
        _01_dist.append(dist(embeddings[0][batch_index], embeddings[1][batch_index]).item())
        _02_dist.append(dist(embeddings[0][batch_index], embeddings[2][batch_index]).item())
        if math.isnan(_01_dist[-1]):
            print('nan: {}, {}'.format(embeddings[0][batch_index], embeddings[1][batch_index]))
        if math.isnan(_02_dist[-1]):
            print('nan: {}, {}'.format(embeddings[0][batch_index], embeddings[2][batch_index]))

    if i % 10 == 0:
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

