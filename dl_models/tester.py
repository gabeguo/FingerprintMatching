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

from common_filepaths import DATA_FOLDER, SUBSET_DATA_FOLDER

MODEL_PATH = '/data/therealgabeguo/embedding_net_weights.pth'

batch_size=8

the_data_folder = DATA_FOLDER

test_dataset = TripletDataset(FingerprintDataset(os.path.join(the_data_folder, 'test'), train=False))
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

SAME_PERSON = 0
DIFF_PERSON = 1
same_sensor_dist = {SAME_PERSON : list(), DIFF_PERSON : list()}
diff_sensor_dist = {SAME_PERSON : list(), DIFF_PERSON : list()}
same_finger_dist = {SAME_PERSON : list(), DIFF_PERSON : list()}
diff_finger_dist = {SAME_PERSON : list(), DIFF_PERSON : list()}

# finger_to_finger_dist[i][j][k] = list of distances between FGRP (i, j) for matching status k (SAME_PERSON or DIFF_PERSON)
finger_to_finger_dist = [[[[] for k in (SAME_PERSON, DIFF_PERSON)] for j in range(0, 10+1)] for i in range(0, 10+1)]

# Pre: parameters are 2 1D tensors
def euclideanDist(tensor1, tensor2):
    return (tensor1 - tensor2).pow(2).sum(0)

# Inputs: (_01_dist, _02_dist) are distance between anchor and (positive, negative), repsectively
# Returns: (accuracies, fpr, tpr, roc_auc, threshold)
# - accuracies are at every possible threshold
# - fpr is false positive rate at every possible threshold (padded with 0 and 1 at end)
# - tpr is true positive rate at every possible threshold (padded with 0 and 1 at end)
# - roc_auc is scalar: area under fpr (x-axis) vs tpr (y-axis) curve
# - threshold is scalar: below this distance, fingerpritnts match; above, they don't match
def get_metrics(_01_dist, _02_dist):
    all_distances = _01_dist +_02_dist
    all_distances.sort()

    tp, fp, tn, fn = list(), list(), list(), list()
    acc = list()

    # try different thresholds
    for dist in all_distances:
        tp.append(len([x for x in _01_dist if x < dist]))
        tn.append(len([x for x in _02_dist if x >= dist]))
        fn.append(len(_01_dist) - tp[-1])
        fp.append(len(_02_dist) - tn[-1])

        acc.append((tp[-1] + tn[-1]) / len(all_distances))
    threshold = all_distances[max(range(len(acc)), key=acc.__getitem__)]

    # ROC AUC is FPR = FP / (FP + TN) (x-axis) vs TPR = TP / (TP + FN) (y-axis)
    fpr = [0] + [fp[i] / (fp[i] + tn[i]) for i in range(len(fp))] + [1]
    tpr = [0] + [tp[i] / (tp[i] + fn[i]) for i in range(len(tp))] + [1]
    auc = sum([tpr[i] * (fpr[i] - fpr[i - 1]) for i in range(1, len(tpr))])

    return acc, fpr, tpr, auc, threshold

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
        anchor_fgrp, pos_fgrp, neg_fgrp = int(get_fgrp(anchor_filename)), int(get_fgrp(pos_filename)), int(get_fgrp(neg_filename))
        anchor_sensor, pos_sensor, neg_sensor = get_sensor(anchor_filename), get_sensor(pos_filename), get_sensor(neg_filename)

        # print(anchor_filename, pos_filename, neg_filename)

        assert get_id(anchor_filename) == get_id(pos_filename)
        assert get_id(anchor_filename) != get_id(neg_filename)

        # TODO: finish this
        finger_to_finger_dist[anchor_fgrp][pos_fgrp][SAME_PERSON].append(_01_dist[-1])
        finger_to_finger_dist[pos_fgrp][anchor_fgrp][SAME_PERSON].append(_01_dist[-1])
        finger_to_finger_dist[anchor_fgrp][neg_fgrp][DIFF_PERSON].append(_02_dist[-1])
        finger_to_finger_dist[neg_fgrp][anchor_fgrp][DIFF_PERSON].append(_02_dist[-1])

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

# CALCULATE ACCURACY AND ROC AUC
accs, fpr, tpr, auc, threshold = get_metrics(_01_dist, _02_dist)

max_acc = max(accs)
print('best accuracy:', max_acc)
"""
import matplotlib.pyplot as plt
plt.plot([i for i in range(len(acc))], acc)
plt.show()
"""

print('auc = {}'.format(auc))
import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
assert auc >= 0 and auc <= 1

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
roc_by_trait = dict()

# distance by sample trait (sensor type, finger)
for trait_name, the_dists in zip(['same sensor', 'diff sensor', 'same finger', 'diff finger'], \
                            [same_sensor_dist, diff_sensor_dist, same_finger_dist, diff_finger_dist]):
    print('Results for {}'.format(trait_name))
    for person in (SAME_PERSON, DIFF_PERSON):
        person_str = 'same' if person == SAME_PERSON else 'diff'
        print('\tnum people in category - {} person, {}: {}'.format(person_str, trait_name, len(the_dists[person])))
        print('\t\taverage squared L2 distance between {} person: {}'.format(person_str, np.mean(the_dists[person])))
        print('\t\tstd of  squared L2 distance between {} person: {}'.format(person_str, np.std(the_dists[person])))

    curr_accs, curr_fpr, curr_tpr, curr_roc_auc, curr_threshold = get_metrics(the_dists[SAME_PERSON], the_dists[DIFF_PERSON])
    acc_by_trait[trait_name] = max(curr_accs)
    roc_by_trait[trait_name] = curr_roc_auc

    print('\tacc:', max(curr_accs))
    print('\troc auc:', curr_roc_auc)

    # tp = len([x for x in the_dists[SAME_PERSON] if x < threshold])
    # tn = len([x for x in the_dists[DIFF_PERSON] if x >= threshold])
    # fn = len(the_dists[SAME_PERSON]) - tp
    # fp = len(the_dists[DIFF_PERSON]) - tn
    # acc = (tp + tn) / (len(the_dists[SAME_PERSON]) + len(the_dists[DIFF_PERSON]))
    # print('\tacc:', acc)
    # acc_by_trait[trait_name] = acc

from datetime import datetime
# TODO: fix datae formatting
datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
with open('/data/therealgabeguo/results/test_results_{}.txt'.format(datetime_str), 'w') as fout:
    fout.write('data folder: {}\n\n'.format(the_data_folder))
    fout.write('average squared L2 distance between positive pairs: {}\n'.format(np.mean(_01_dist)))
    fout.write('std of  squared L2 distance between positive pairs: {}\n'.format(np.std(_01_dist)))
    fout.write('average squared L2 distance between negative pairs: {}\n'.format(np.mean(_02_dist)))
    fout.write('std of  squared L2 distance between negative pairs: {}\n'.format(np.std(_02_dist)))
    fout.write('best accuracy: {}\n'.format(str(max_acc)))
    fout.write('ROC AUC: {}\n\n'.format(auc))

    # distance by sample trait (sensor type, finger)
    for trait_name, the_dists in zip(['same sensor', 'diff sensor', 'same finger', 'diff finger'], \
                                [same_sensor_dist, diff_sensor_dist, same_finger_dist, diff_finger_dist]):
        fout.write('Results for {}\n'.format(trait_name))
        for person in (SAME_PERSON, DIFF_PERSON):
            person_str = 'same' if person == SAME_PERSON else 'diff'
            fout.write('\tnum people in category - {} person, {}: {}\n'.format(person_str, trait_name, len(the_dists[person])))
            fout.write('\t\taverage squared L2 distance between {} person: {}\n'.format(person_str, np.mean(the_dists[person])))
            fout.write('\t\tstd of  squared L2 distance between {} person: {}\n'.format(person_str, np.std(the_dists[person])))
        fout.write('\taccuracy: {}\n'.format(str(acc_by_trait[trait_name])))
        fout.write('\troc auc: {}\n\n'.format(str(roc_by_trait[trait_name])))
