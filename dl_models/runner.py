import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import os

sys.path.append('../')

from trainer import *
from losses import *
from siamese_datasets import *
from fingerprint_dataset import *
from embedding_models import *

from common_filepaths import DATA_FOLDER

MODEL_PATH = 'embedding_net_weights.pth'

batch_size=4

training_dataset = TripletDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'train'), train=True))
training_dataset = torch.utils.data.Subset(training_dataset, list(range(0, len(training_dataset), 5)))
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TripletDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'val'), train=True))
val_dataset = torch.utils.data.Subset(val_dataset, list(range(0, len(val_dataset), 5)))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TripletDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'test'), train=True))
test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, len(test_dataset), 5)))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

embedder = EmbeddingNet()
embedder.load_state_dict(torch.load(MODEL_PATH))
# freeze all layers except the last one
"""
for the_param in list(embedder.feature_extractor.children())[:-1]:
    the_param.requires_grad = False
"""
# make triplet net
triplet_net = TripletNet(embedder)

# TRAIN

learning_rate = 0.01
momentum = 0.9
weight_decay = 5e-5
lr_decay_step=3
lr_decay_factor=0.9
optimizer = optim.SGD(triplet_net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)
tripletLoss_margin = 1

fit(train_loader=train_dataloader, val_loader=val_dataloader, model=triplet_net, \
    loss_fn=TripletLoss(margin=tripletLoss_margin), optimizer=optimizer, scheduler=scheduler, \
    n_epochs=1, cuda='cuda:0', log_interval=10, metrics=[], start_epoch=0)

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
        _01_dist.append(dist(embeddings[0][batch_index], embeddings[1][batch_index]).item())
        _02_dist.append(dist(embeddings[0][batch_index], embeddings[2][batch_index]).item())

    #print(test_filepaths, test_labels)

#print(_01_dist[0].size())
#print(_02_dist[0].size())

print('number of testing positive pairs:', len(_01_dist))
print('number of testing negative pairs:', len(_02_dist))

print('average cosine sim between matching pairs:', sum(_01_dist) / len(_01_dist))
print('average cosine sim between non-matching pairs:', sum(_02_dist) / len(_02_dist))
