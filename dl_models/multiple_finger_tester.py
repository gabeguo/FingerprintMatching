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
from fingerprint_dataset import *
from multiple_finger_datasets import *
from embedding_models import *
from fileProcessingUtil import *

from common_filepaths import *

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

    for i in range(1, len(fpr)):
        assert fpr[i] >= fpr[i - 1]
        assert tpr[i] >= tpr[i - 1]

    return acc, fpr, tpr, auc, threshold

# Create output directory
from datetime import datetime
datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
output_dir = os.path.join('/data/therealgabeguo/results', datetime_str)
os.makedirs(output_dir, exist_ok=True)

# Model weights
MODEL_PATH = '/data/therealgabeguo/embedding_net_weights.pth'

# CUDA
CUDA = 'cuda:2'

# Data loading 
batch_size=1 # must be 1

num_anchors=4
num_pos=4
num_neg=4

the_data_folder = BALANCED_DATA_FOLDER
test_dataset = MultipleFingerDataset(FingerprintDataset(os.path.join(the_data_folder, 'test'), train=False), num_anchors, num_pos, num_neg)
print('loaded test dataset: {}'.format(the_data_folder))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# CREATE EMBEDDER
embedder = EmbeddingNet()

# distances between embedding of positive and negative pair
_01_dist = []
_02_dist = []

# LOAD MODEL

embedder.load_state_dict(torch.load(MODEL_PATH))
embedder.eval()
embedder.to(CUDA)

# TEST
assert batch_size == 1

data_iter = iter(test_dataloader)
for i in range(len(test_dataloader)):
    test_images, test_labels, test_filepaths = next(data_iter)

    # test_images is 3 (anchor, pos, neg) * N (number of sample images) * image_size (1*3*224*224)

    curr_anchor_pos_dists = []
    curr_anchor_neg_dists = []

    for i_a in range(num_anchors):
        curr_anchor = test_images[0][i_a].to(CUDA)
        embedding_anchor = torch.flatten(embedder(curr_anchor))
        assert len(embedding_anchor.size()) == 1 and embedding_anchor.size(dim=0) == 512
        for i_p in range(num_pos):
            curr_pos = test_images[1][i_p].to(CUDA)
            embedding_pos = torch.flatten(embedder(curr_pos))
            assert len(embedding_pos.size()) == 1 and embedding_pos.size(dim=0) == 512
            curr_anchor_pos_dists.append(euclideanDist(embedding_anchor, embedding_pos).item())
        for i_n in range(num_neg):
            curr_neg = test_images[2][i_n].to(CUDA)
            embedding_neg = torch.flatten(embedder(curr_neg))
            assert len(embedding_neg.size()) == 1 and embedding_neg.size(dim=0) == 512
            curr_anchor_neg_dists.append(euclideanDist(embedding_anchor, embedding_neg).item())

    _01_dist.append(np.mean(curr_anchor_pos_dists))
    _02_dist.append(np.mean(curr_anchor_neg_dists))


    if i % 100 == 0:
        print('Batch (item) {} out of {}'.format(i, len(test_dataloader)))
        print('\taverage, std squared L2 distance between positive pairs {:.3f}, {:.3f}'.format(np.mean(_01_dist), np.std(_01_dist)))
        print('\taverage, std squared L2 distance between negative pairs {:.3f}, {:.3f}'.format(np.mean(_02_dist), np.std(_02_dist)))

# CALCULATE ACCURACY AND ROC AUC
accs, fpr, tpr, auc, threshold = get_metrics(_01_dist, _02_dist)

max_acc = max(accs)
print('best accuracy:', max_acc)

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
assert auc >= 0 and auc <= 1

# PRINT DISTANCES
_01_dist = np.array(_01_dist)
_02_dist = np.array(_02_dist)

print('Number anchor, pos, neg fingers: {}, {}, {}'.format(num_anchors, num_pos, num_neg))

print('number of testing positive pairs:', len(_01_dist))
print('number of testing negative pairs:', len(_02_dist))

print('average squared L2 distance between positive pairs:', np.mean(_01_dist))
print('std of  squared L2 distance between positive pairs:', np.std(_01_dist))
print('average squared L2 distance between negative pairs:', np.mean(_02_dist))
print('std of  squared L2 distance between negative pairs:', np.std(_02_dist))

# do the output

results_fname = os.path.join(output_dir, 'test_results_multi_finger.txt')
with open(results_fname, 'w') as fout:
    fout.write('data folder: {}\n\n'.format(the_data_folder))
    fout.write('Number anchor, pos, neg fingers: {}, {}, {}\n\n'.format(num_anchors, num_pos, num_neg))
    fout.write('average squared L2 distance between positive pairs: {}\n'.format(np.mean(_01_dist)))
    fout.write('std of  squared L2 distance between positive pairs: {}\n'.format(np.std(_01_dist)))
    fout.write('average squared L2 distance between negative pairs: {}\n'.format(np.mean(_02_dist)))
    fout.write('std of  squared L2 distance between negative pairs: {}\n'.format(np.std(_02_dist)))
    fout.write('best accuracy: {}\n'.format(str(max_acc)))
    fout.write('ROC AUC: {}\n\n'.format(auc))