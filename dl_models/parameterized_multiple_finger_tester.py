import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import sys
import os
import math
import argparse
from scipy.stats import ttest_ind

import json

sys.path.append('../')
sys.path.append('../directory_organization')

import matplotlib
matplotlib.use("Agg")

from trainer import *
from fingerprint_dataset import *
from multiple_finger_datasets import *
from embedding_models import *
from fileProcessingUtil import *

from common_filepaths import *

# Default Values
DEFAULT_OUTPUT_ROOT = '/data/therealgabeguo/results'
DEFAULT_CUDA = 'cuda:2'

# JSON Keys
DATASET_KEY = 'dataset'
WEIGHTS_KEY = 'weights'
CUDA_KEY = 'cuda'
OUTPUT_DIR_KEY = 'output dir'
NUM_ANCHOR_KEY = 'number anchor fingers per set'
SCALE_FACTOR_KEY = 'number of times looped through dataset'
NUM_POS_KEY = 'number positive fingers per set'
NUM_NEG_KEY = 'number negative fingers per set'
NUM_SAMPLES_KEY = 'number of distinct samples in dataset'
NUM_POS_PAIRS_KEY = 'number of positive pairs'
NUM_NEG_PAIRS_KEY = 'number of negative pairs'
MEAN_POS_DIST_KEY = 'average squared L2 distance between positive pairs'
STD_POS_DIST_KEY = 'std of  squared L2 distance between positive pairs'
MEAN_NEG_DIST_KEY = 'average squared L2 distance between negative pairs'
STD_NEG_DIST_KEY = 'std of  squared L2 distance between negative pairs'
ACC_KEY = 'best accuracy'
ROC_AUC_KEY = 'ROC AUC'
T_VAL_KEY = 'Welch\'s t'
P_VAL_KEY = 'p-value'

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
# - welch_t is value of Welch's two-sample t-test between same-person and diff-person pairs
# - p-val is statistical significance
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

    # Welch's t-test
    welch_t, p_val = ttest_ind(_01_dist, _02_dist, equal_var=False)

    return acc, fpr, tpr, auc, threshold, welch_t, p_val

def create_output_dir(output_root):
    from datetime import datetime
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(output_root, datetime_str)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Data loading 
batch_size=1 # must be 1

def main(the_data_folder, weights_path, cuda, output_dir, num_anchors, num_pos, num_neg, scale_factor=1):
    print('Number anchor, pos, neg fingers: {}, {}, {}'.format(num_anchors, num_pos, num_neg))

    fingerprint_dataset = FingerprintDataset(os.path.join(the_data_folder, 'test'), train=False)
    test_dataset = MultipleFingerDataset(fingerprint_dataset, num_anchors, num_pos, num_neg, SCALE_FACTOR=scale_factor)
    print('loaded test dataset: {}'.format(the_data_folder))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    dataset_name = the_data_folder[:-1 if the_data_folder[-1] == '/' else len(the_data_folder)].split('/')[-1]
    print('dataset name:', dataset_name)

    weights_name = (weights_path.split('/')[-1])[:-4]
    print('weights name:', weights_name)

    # CREATE EMBEDDER
    embedder = EmbeddingNet()

    # distances between embedding of positive and negative pair
    _01_dist = []
    _02_dist = []

    # LOAD MODEL

    embedder.load_state_dict(torch.load(weights_path))
    embedder.eval()
    embedder.to(cuda)

    # TEST
    assert batch_size == 1

    data_iter = iter(test_dataloader)
    for i in range(len(test_dataloader)):
        test_images, test_labels, test_filepaths = next(data_iter)

        # test_images is 3 (anchor, pos, neg) * N (number of sample images) * image_size (1*3*224*224)

        curr_anchor_pos_dists = []
        curr_anchor_neg_dists = []

        for i_a in range(num_anchors):
            curr_anchor = test_images[0][i_a].to(cuda)
            embedding_anchor = torch.flatten(embedder(curr_anchor))
            assert len(embedding_anchor.size()) == 1 and embedding_anchor.size(dim=0) == 512
            for i_p in range(num_pos):
                curr_pos = test_images[1][i_p].to(cuda)
                embedding_pos = torch.flatten(embedder(curr_pos))
                assert len(embedding_pos.size()) == 1 and embedding_pos.size(dim=0) == 512
                curr_anchor_pos_dists.append(euclideanDist(embedding_anchor, embedding_pos).item())
            for i_n in range(num_neg):
                curr_neg = test_images[2][i_n].to(cuda)
                embedding_neg = torch.flatten(embedder(curr_neg))
                assert len(embedding_neg.size()) == 1 and embedding_neg.size(dim=0) == 512
                curr_anchor_neg_dists.append(euclideanDist(embedding_anchor, embedding_neg).item())

        _01_dist.append(np.mean(curr_anchor_pos_dists))
        _02_dist.append(np.mean(curr_anchor_neg_dists))

        if i % 100 == 0:
            print('Batch (item) {} out of {}'.format(i, len(test_dataloader)))
            print('\taverage, std squared L2 distance between positive pairs {:.3f}, {:.3f}'.format(np.mean(_01_dist), np.std(_01_dist)))
            print('\taverage, std squared L2 distance between negative pairs {:.3f}, {:.3f}'.format(np.mean(_02_dist), np.std(_02_dist)))

        # TODO: update finger-by-finger

    # CALCULATE ACCURACY AND ROC AUC
    accs, fpr, tpr, auc, threshold, welch_t, p_val = get_metrics(_01_dist, _02_dist)

    max_acc = max(accs)
    print('best accuracy:', max_acc)

    print('auc = {}'.format(auc))
    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'roc_curve_{}_{}_{}_{}_{}.pdf'.format(\
        dataset_name, weights_name, num_anchors, num_pos, num_neg)))
    plt.savefig(os.path.join(output_dir, 'roc_curve_{}_{}_{}_{}_{}.png'.format(\
        dataset_name, weights_name, num_anchors, num_pos, num_neg)))
    plt.clf(); plt.close()
    assert auc >= 0 and auc <= 1

    # do the output

    final_results = {
        DATASET_KEY: the_data_folder,
        WEIGHTS_KEY: weights_path,
        CUDA_KEY: cuda,
        OUTPUT_DIR_KEY: output_dir,
        NUM_ANCHOR_KEY: num_anchors,
        SCALE_FACTOR_KEY: scale_factor,
        NUM_POS_KEY: num_pos,
        NUM_NEG_KEY: num_neg,
        NUM_SAMPLES_KEY: len(fingerprint_dataset),
        NUM_POS_PAIRS_KEY: len(_01_dist),
        NUM_NEG_PAIRS_KEY: len(_02_dist),
        MEAN_POS_DIST_KEY: np.mean(_01_dist),
        STD_POS_DIST_KEY: np.std(_01_dist),
        MEAN_NEG_DIST_KEY: np.mean(_02_dist),
        STD_NEG_DIST_KEY: np.std(_02_dist),
        ACC_KEY: max_acc,
        ROC_AUC_KEY: auc,
        T_VAL_KEY: welch_t,
        P_VAL_KEY: p_val
    }

    results_fname = os.path.join(output_dir, \
        'test_results_{}_{}_{}_{}_{}.json'.format(dataset_name, weights_name, num_anchors, num_pos, num_neg))
    with open(results_fname, 'w') as fout:
        fout.write(json.dumps(final_results, indent=4))

    # print output
    with open(results_fname, 'r') as fin:
        for line in fin:
            print(line.rstrip())
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser('parameterized_multiple_finger_tester.py')
    parser.add_argument('--dataset', '-d', help='Path to folders containing images', type=str)
    parser.add_argument('--weights', '-w', help='Path to model weights', type=str)
    parser.add_argument('--cuda', '-c', nargs='?', \
        const=DEFAULT_CUDA, default=DEFAULT_CUDA, help='Name of GPU we want to use', type=str)
    parser.add_argument('--num_fingers', '-n', nargs='?', help='Number of fingers to test', \
        const=1, default=1, type=int)
    parser.add_argument('--output_root', '-o', nargs='?', help='Root directory for output', \
        const=DEFAULT_OUTPUT_ROOT, default=DEFAULT_OUTPUT_ROOT, type=str)
    parser.add_argument('--scale_factor', '-s', nargs='?', help='Number of times to loop through the dataset to create triplets', \
        const=1, default=1, type=int)

    args = parser.parse_args()

    dataset = args.dataset
    weights = args.weights
    cuda = args.cuda
    num_fingers = args.num_fingers
    output_dir = create_output_dir(args.output_root)
    scale_factor = args.scale_factor

    assert num_fingers > 0
    assert scale_factor >= 1

    main(the_data_folder=dataset, weights_path=weights, cuda=cuda, output_dir=output_dir, \
        num_anchors=num_fingers, num_pos=num_fingers, num_neg=num_fingers, scale_factor=scale_factor)