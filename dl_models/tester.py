import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import os
import math

sys.path.append('../')
sys.path.append('../directory_organization')

from trainer import *
from losses import *
from siamese_datasets import *
from fingerprint_dataset import *
from embedding_models import *
from fileProcessingUtil import *

from common_filepaths import DATA_FOLDER

MODEL_PATH = 'embedding_net_weights.pth'

batch_size=8

training_dataset = TripletDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'train'), train=True))
#training_dataset = torch.utils.data.Subset(training_dataset, list(range(0, len(training_dataset), 5)))
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TripletDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'val'), train=False))
#val_dataset = torch.utils.data.Subset(val_dataset, list(range(0, len(val_dataset), 5)))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TripletDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'test'), train=False))
#test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, len(test_dataset), 100)))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# SHOW IMAGES
"""
import matplotlib.pyplot as plt
it = iter(val_dataloader)
for i in range(5):
    images, labels, filepaths = next(it)
    next_img = images[2][0]
    the_min = torch.min(next_img)
    the_max = torch.max(next_img)
    next_img = (next_img - the_min) / (the_max - the_min)
    print(next_img[0])
    plt.imshow(next_img.permute(1, 2, 0))
    plt.show()
"""

# CREATE EMBEDDER

embedder = EmbeddingNet()

# CREATE TRIPLET NET
triplet_net = TripletNet(embedder)

# distances between embedding of positive and negative pair
_01_dist = []
_02_dist = []

SAME_PERSON = 1
DIFF_PERSON = 0
same_sensor_dist = {SAME_PERSON : list(), DIFF_PERSON : list()}
diff_sensor_dist = {SAME_PERSON : list(), DIFF_PERSON : list()}
same_finger_dist = {SAME_PERSON : list(), DIFF_PERSON : list()}
diff_finger_dist = {SAME_PERSON : list(), DIFF_PERSON : list()}

# Pre: parameters are 2 1D tensors
def euclideanDist(tensor1, tensor2):
    return (tensor1 - tensor2).pow(2).sum(0)


# LOAD MODEL
embedder.load_state_dict(torch.load(MODEL_PATH))
embedder.eval()
embedder = embedder.to('cuda:1')

# TEST

for i in range(len(test_dataloader)):
    test_images, test_labels, test_filepaths = next(iter(test_dataloader))

    test_images = [item.to('cuda:1') for item in test_images]

    embeddings = [torch.reshape(e, (batch_size, e.size()[1])) for e in triplet_net(*test_images)]
    # len(embeddings) == 3 reprenting the following (anchor, pos, neg)
    # Each index in the list contains a tensor of size (batch size, embedding length)

    for batch_index in range(batch_size):
        _01_dist.append(euclideanDist(embeddings[0][batch_index], embeddings[1][batch_index]).item())
        _02_dist.append(euclideanDist(embeddings[0][batch_index], embeddings[2][batch_index]).item())

        # process traits of these samples (same finger, same sensor)
        anchor_filename, pos_filename, neg_filename = \
            test_filepaths[0][batch_index], test_filepaths[1][batch_index], test_filepaths[2][batch_index]
        anchor_filename, pos_filename, neg_filename = anchor_filename.split('/')[-1], pos_filename.split('/')[-1], neg_filename.split('/')[-1]
        anchor_fgrp, pos_fgrp, neg_fgrp = get_fgrp(anchor_filename), get_fgrp(pos_filename), get_fgrp(neg_filename)
        anchor_sensor, pos_sensor, neg_sensor = get_sensor(anchor_filename), get_sensor(pos_filename), get_sensor(neg_filename)

        # print(anchor_filename, pos_filename, neg_filename)

        assert get_id(anchor_filename) == get_id(pos_filename)
        assert get_id(anchor_filename) != get_id(neg_filename)

        # same finger, same person
        if anchor_fgrp == pos_fgrp:
            same_finger_dist[SAME_PERSON].append(_01_dist[-1])
        else:
            diff_finger_dist[SAME_PERSON].append(_01_dist[-1])
        # same finger, diff person
        if anchor_fgrp == neg_fgrp:
            same_finger_dist[DIFF_PERSON].append(_02_dist[-1]) #_02_dist[-1] is the dist between the current anchor and negative sample
        else:
            diff_finger_dist[DIFF_PERSON].append(_02_dist[-1])
        # same sensor, same person
        if anchor_sensor == pos_sensor:
            same_sensor_dist[SAME_PERSON].append(_01_dist[-1])
        else:
            diff_sensor_dist[SAME_PERSON].append(_01_dist[-1])
        # same sensor, diff person
        if anchor_sensor == neg_sensor:
            same_sensor_dist[DIFF_PERSON].append(_02_dist[-1])
        else:
            diff_sensor_dist[DIFF_PERSON].append(_02_dist[-1])

    if i % 40 == 0:
        print('Batch {} out of {}'.format(i, len(test_dataloader)))
        print('\taverage squared L2 distance between positive pairs:', np.mean(np.array(_01_dist)))
        print('\taverage squared L2 distance between negative pairs:', np.mean(np.array(_02_dist)))

# FIND THRESHOLDS
all_distances = _01_dist +_02_dist
all_distances.sort()

tp, fp, tn, fn = list(), list(), list(), list()
acc = list()

for dist in all_distances:
    tp.append(len([x for x in _01_dist if x < dist]))
    tn.append(len([x for x in _02_dist if x >= dist]))
    fn.append(len(_01_dist) - tp[-1])
    fp.append(len(_02_dist) - tn[-1])

    acc.append((tp[-1] + tn[-1]) / len(all_distances))

max_acc = max(acc)
print('best accuracy:', max(acc))
threshold = all_distances[max(range(len(acc)), key=acc.__getitem__)]
"""
import matplotlib.pyplot as plt
plt.plot([i for i in range(len(acc))], acc)
plt.show()
"""
# PRINT DISTANCES
_01_dist = np.array(_01_dist)
_02_dist = np.array(_02_dist)

print('number of testing positive pairs:', len(_01_dist))
print('number of testing negative pairs:', len(_02_dist))

print('average squared L2 distance between positive pairs:', np.mean(_01_dist))
print('std of  squared L2 distance between positive pairs:', np.std(_01_dist))
print('average squared L2 distance between negative pairs:', np.mean(_02_dist))
print('std of  squared L2 distance between negative pairs:', np.std(_02_dist))

acc_by_trait = dict()

# distance by sample trait (sensor type, finger)
for trait_name, the_dists in zip(['same sensor', 'diff sensor', 'same finger', 'diff finger'], \
                            [same_sensor_dist, diff_sensor_dist, same_finger_dist, diff_finger_dist]):
    print('Results for {}'.format(trait_name))
    for person in (SAME_PERSON, DIFF_PERSON):
        person_str = 'same' if person == SAME_PERSON else 'diff'
        print('\tnum people in category - {} person, {}: {}'.format(person_str, trait_name, len(the_dists[person])))
        print('\t\taverage squared L2 distance between {} person: {}'.format(person_str, np.mean(the_dists[person])))
        print('\t\tstd of  squared L2 distance between {} person: {}'.format(person_str, np.std(the_dists[person])))

    tp = len([x for x in the_dists[SAME_PERSON] if x < threshold])
    tn = len([x for x in the_dists[DIFF_PERSON] if x >= threshold])
    fn = len(the_dists[SAME_PERSON]) - tp
    fp = len(the_dists[DIFF_PERSON]) - tn

    acc = (tp + tn) / (len(the_dists[SAME_PERSON]) + len(the_dists[DIFF_PERSON]))
    print('\tacc:', acc)

    acc_by_trait[trait_name] = acc

from datetime import datetime
datetime_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
with open('results/test_results_{}.txt'.format(datetime_str), 'w') as fout:
    fout.write('average squared L2 distance between positive pairs: {}\n'.format(np.mean(_01_dist)))
    fout.write('std of  squared L2 distance between positive pairs: {}\n'.format(np.std(_01_dist)))
    fout.write('average squared L2 distance between negative pairs: {}\n'.format(np.mean(_02_dist)))
    fout.write('std of  squared L2 distance between negative pairs: {}\n'.format(np.std(_02_dist)))
    fout.write('best accuracy: {}\n\n'.format(str(max_acc)))

    # distance by sample trait (sensor type, finger)
    for trait_name, the_dists in zip(['same sensor', 'diff sensor', 'same finger', 'diff finger'], \
                                [same_sensor_dist, diff_sensor_dist, same_finger_dist, diff_finger_dist]):
        fout.write('Results for {}\n'.format(trait_name))
        for person in (SAME_PERSON, DIFF_PERSON):
            person_str = 'same' if person == SAME_PERSON else 'diff'
            fout.write('\tnum people in category - {} person, {}: {}\n'.format(person_str, trait_name, len(the_dists[person])))
            fout.write('\t\taverage squared L2 distance between {} person: {}\n'.format(person_str, np.mean(the_dists[person])))
            fout.write('\t\tstd of  squared L2 distance between {} person: {}\n'.format(person_str, np.std(the_dists[person])))
        fout.write('\taccuracy: {}\n\n'.format(str(acc_by_trait[trait_name])))
