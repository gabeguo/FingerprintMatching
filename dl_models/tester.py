import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
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

from common_filepaths import DATA_FOLDER, SUBSET_DATA_FOLDER, BALANCED_DATA_FOLDER, UNSEEN_DATA_FOLDER, EXTRA_DATA_FOLDER, ENHANCED_DATA_FOLDER

# Create output directory
from datetime import datetime
datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
output_dir = os.path.join('/data/therealgabeguo/results', datetime_str)
os.makedirs(output_dir, exist_ok=True)

# Model weights
MODEL_PATH = '/data/therealgabeguo/embedding_net_weights.pth'

# Data loading 
batch_size=8
the_data_folder = ENHANCED_DATA_FOLDER
test_dataset = TripletDataset(FingerprintDataset(os.path.join(the_data_folder, 'test'), train=False))
print('loaded test dataset: {}'.format(the_data_folder))
# test_dataset = torch.utils.data.ConcatDataset(\
#     [TripletDataset(FingerprintDataset(os.path.join(the_data_folder, 'test'), train=False)), \
#     TripletDataset(FingerprintDataset(os.path.join(the_data_folder, 'test'), train=False)), \
#     TripletDataset(FingerprintDataset(os.path.join(the_data_folder, 'test'), train=False))] \
# )
#test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, len(test_dataset), 10)))
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

data_iter = iter(test_dataloader)
for i in range(len(test_dataloader)):
    test_images, test_labels, test_filepaths = next(data_iter)

    test_images = [item.to('cuda:1') for item in test_images]

    #print(triplet_net(*test_images)[0].size())
    embeddings = [torch.reshape(e, (e.size()[0], e.size()[1])) for e in triplet_net(*test_images)]
    # len(embeddings) == 3 reprenting the following (anchor, pos, neg)
    # Each index in the list contains a tensor of size (batch size, embedding length)

    for batch_index in range(embeddings[0].size()[0]): # should be equivalent to range(batch_size):
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
plt.title('ROC Curve')
plt.grid()
plt.savefig(os.path.join(output_dir, 'roc_curve.pdf'))
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.clf(); plt.close()
#plt.show()
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
finger_to_finger_num_people = np.zeros((10, 10))
finger_to_finger_percent_samePerson = np.zeros((10, 10))
finger_to_finger_acc = np.zeros((10, 10))
finger_to_finger_roc = np.zeros((10, 10))
f2f_meanDist_samePerson = np.zeros((10, 10))
f2f_meanDist_diffPerson = np.zeros((10, 10))
f2f_stdDist_samePerson = np.zeros((10, 10))
f2f_stdDist_diffPerson = np.zeros((10, 10))

# stats by finger-finger-pair
fgrp_names = ['Right Thumb', 'Right Index', 'Right Middle', 'Right Ring', 'Right Pinky', \
            'Left Thumb', 'Left Index', 'Left Middle', 'Left Ring', 'Left Pinky']
for i in range(1, 10+1):
    for j in range(1, 10+1):
        same_person_dists = finger_to_finger_dist[i][j][SAME_PERSON]
        diff_person_dists = finger_to_finger_dist[i][j][DIFF_PERSON]

        curr_accs, curr_fpr, curr_tpr, curr_roc_auc, curr_threshold = get_metrics(same_person_dists, diff_person_dists)
        
        _i, _j = i - 1, j - 1
        finger_to_finger_acc[_i][_j] = max(curr_accs)
        finger_to_finger_roc[_i][_j] = curr_roc_auc

        finger_to_finger_num_people[_i][_j] = len(same_person_dists) + len(diff_person_dists)
        finger_to_finger_percent_samePerson[_i][_j] = len(same_person_dists) / finger_to_finger_num_people[_i][_j]

        f2f_meanDist_samePerson[_i][_j] = np.mean(same_person_dists)
        f2f_meanDist_diffPerson[_i][_j] = np.mean(diff_person_dists)
        f2f_stdDist_samePerson[_i][_j] = np.std(same_person_dists)
        f2f_stdDist_diffPerson[_i][_j] = np.std(diff_person_dists)

print('Accuracy finger by finger:')
print(np.array_str(finger_to_finger_acc, precision=3, suppress_small=True))
print('ROC AUC finger by finger:')
print(np.array_str(finger_to_finger_roc, precision=3, suppress_small=True))
print('Number of finger-to-finger pairs:')
print(np.array_str(finger_to_finger_num_people, suppress_small=True))
print('Proportion of same-person samples by finger combo:')
print(np.array_str(finger_to_finger_percent_samePerson, precision=3, suppress_small=True))

import seaborn as sns

plt.subplots_adjust(bottom=0.22, left=0.22)
plt.title('Finger-to-Finger Accuracy')
sns.heatmap(finger_to_finger_acc.round(3), annot=True, xticklabels=fgrp_names, yticklabels=fgrp_names, cmap='Reds')
plt.savefig(os.path.join(output_dir, 'acc.pdf'))
plt.savefig(os.path.join(output_dir, 'acc.png'))
plt.clf(); plt.close()

plt.subplots_adjust(bottom=0.22, left=0.22)
plt.title('Finger-to-Finger ROC AUC')
sns.heatmap(finger_to_finger_roc.round(3), annot=True, xticklabels=fgrp_names, yticklabels=fgrp_names, cmap='Reds')
plt.savefig(os.path.join(output_dir, 'roc_auc.pdf'))
plt.savefig(os.path.join(output_dir, 'roc_auc.png'))
plt.clf(); plt.close()

plt.subplots_adjust(bottom=0.22, left=0.22)
plt.title('Number of Finger-to-Finger Pairs')
sns.heatmap(finger_to_finger_num_people, annot=True, xticklabels=fgrp_names, yticklabels=fgrp_names, cmap='Blues', fmt='g')
plt.savefig(os.path.join(output_dir, 'sample_size.pdf'))
plt.savefig(os.path.join(output_dir, 'sample_size.png'))
plt.clf(); plt.close()

plt.subplots_adjust(bottom=0.22, left=0.22)
plt.title('Proportion of Same-Person Samples')
sns.heatmap(finger_to_finger_percent_samePerson.round(3), annot=True, xticklabels=fgrp_names, yticklabels=fgrp_names, cmap='Greens', vmin=0, vmax=1)
plt.savefig(os.path.join(output_dir, 'sample_dist.pdf'))
plt.savefig(os.path.join(output_dir, 'sample_dist.png'))
plt.clf(); plt.close()

# stats by sample trait (sensor type, finger)
for trait_name, the_dists in zip(['same sensor encounter', 'diff sensor encounter', 'same finger', 'diff finger'], \
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

# do the output

results_fname = os.path.join(output_dir, 'test_results.txt')
with open(results_fname, 'w') as fout:
    fout.write('data folder: {}\n\n'.format(the_data_folder))
    fout.write('average squared L2 distance between positive pairs: {}\n'.format(np.mean(_01_dist)))
    fout.write('std of  squared L2 distance between positive pairs: {}\n'.format(np.std(_01_dist)))
    fout.write('average squared L2 distance between negative pairs: {}\n'.format(np.mean(_02_dist)))
    fout.write('std of  squared L2 distance between negative pairs: {}\n'.format(np.std(_02_dist)))
    fout.write('best accuracy: {}\n'.format(str(max_acc)))
    fout.write('ROC AUC: {}\n\n'.format(auc))

    fout.write('Accuracy finger by finger:\n')
    fout.write(np.array_str(finger_to_finger_acc, precision=3, suppress_small=True) + '\n')
    fout.write('ROC AUC finger by finger:\n')
    fout.write(np.array_str(finger_to_finger_roc, precision=3, suppress_small=True) + '\n')
    fout.write('Number of finger-to-finger pairs:\n')
    fout.write(np.array_str(finger_to_finger_num_people, suppress_small=True) + '\n')
    fout.write('Proportion of same-person pairs by finger combo:\n')
    fout.write(np.array_str(finger_to_finger_percent_samePerson, precision=3, suppress_small=True) + '\n\n')

    # distance by sample trait (sensor type, finger)
    for trait_name, the_dists in zip(['same sensor encounter', 'diff sensor encounter', 'same finger', 'diff finger'], \
                                [same_sensor_dist, diff_sensor_dist, same_finger_dist, diff_finger_dist]):
        fout.write('Results for {}\n'.format(trait_name))
        for person in (SAME_PERSON, DIFF_PERSON):
            person_str = 'same' if person == SAME_PERSON else 'diff'
            fout.write('\tnum people in category - {} person, {}: {}\n'.format(person_str, trait_name, len(the_dists[person])))
            fout.write('\t\taverage squared L2 distance between {} person: {}\n'.format(person_str, np.mean(the_dists[person])))
            fout.write('\t\tstd of  squared L2 distance between {} person: {}\n'.format(person_str, np.std(the_dists[person])))
        fout.write('\taccuracy: {}\n'.format(str(acc_by_trait[trait_name])))
        fout.write('\troc auc: {}\n\n'.format(str(roc_by_trait[trait_name])))

