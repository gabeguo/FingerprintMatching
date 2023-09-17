"""
-> Tests the performance of any given model on any given dataset.
-> Can customize number of fingers in each testing set (e.g., anchor set, positive set, negative set).
-> Forces different sensor and different finger.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import sys
import os
import subprocess
import math
import matplotlib.pyplot as plt
import seaborn as sns
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
DEFAULT_OUTPUT_ROOT = '/data/verifiedanivray/results'
DEFAULT_CUDA = 'cuda:2'

# Batch Size constant
batch_size=1 # must be 1
assert batch_size == 1

# JSON Keys
DATASET_KEY = 'dataset'
WEIGHTS_KEY = 'weights'
CUDA_KEY = 'cuda'
OUTPUT_DIR_KEY = 'output dir'
NUM_ANCHOR_KEY = 'number anchor fingers per set'
NUM_POS_KEY = 'number positive fingers per set'
NUM_NEG_KEY = 'number negative fingers per set'
SCALE_FACTOR_KEY = 'number of times looped through dataset'
DIFF_FINGER_CROSS_SET_KEY = 'different fingers across sets'
DIFF_FINGER_WITHIN_SET_KEY = 'different fingers within sets'
DIFF_SENSOR_CROSS_SET_KEY = 'different sensors across sets'
SAME_SENSOR_WITHIN_SET_KEY = 'same sensor within set'
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
DF_KEY = 'degrees of freedom (Welch)'
TP_NAMES_KEY = 'some true positives'
FN_NAMES_KEY = 'some false negatives'
TN_NAMES_KEY = 'some true negatives'
FP_NAMES_KEY = 'some false positives'
TP_NUM_KEY = 'num true positives'
FN_NUM_KEY = 'num false negatives'
TN_NUM_KEY = 'num true negatives'
FP_NUM_KEY = 'num false positives'
BEST_ACC_THRESHOLD_KEY = 'best accuracy threshold'
VALID_FINGERS_KEY = 'fingers used'

# More constants
SAME_PERSON = 0
DIFF_PERSON = 1

# examples needed constant (for confusion diagram)
NUM_EXAMPLES_NEEDED = 10

# Pre: parameters are 2 1D tensors
def euclideanDist(tensor1, tensor2):
    return (tensor1 - tensor2).pow(2).sum(0)

"""
Inputs: (_01_dist, _02_dist) are distance between anchor and (positive, negative), repsectively
Returns: (acccuracies, fpr, tpr, ROC AUC, threshold, welch_t, p_val)
- accuracies are at every possible threshold
- fpr is false positive rate at every possible threshold (padded with 0 and 1 at end)
- tpr is true positive rate at every possible threshold (padded with 0 and 1 at end)
- roc_auc is scalar: area under fpr (x-axis) vs tpr (y-axis) curve
- threshold is scalar: below this distance, fingerpritnts match; above, they don't match
- welch_t is value of Welch's two-sample t-test between same-person and diff-person pairs
- p-val is statistical significance
"""
def get_metrics(_01_score, _02_score):
    all_scores = list(set(_01_score + _02_score))
    all_scores.sort(reverse=True)

    tp, fp, tn, fn = list(), list(), list(), list()
    acc = list()

    # try different thresholds
    for score in all_scores:
        tp.append(len([x for x in _01_score if x > score]))
        tn.append(len([x for x in _02_score if x <= score]))
        fn.append(len(_01_score) - tp[-1])
        fp.append(len(_02_score) - tn[-1])

        acc.append((tp[-1] + tn[-1]) / len(all_scores))
    threshold = all_scores[max(range(len(acc)), key=acc.__getitem__)]

    # ROC AUC is FPR = FP / (FP + TN) (x-axis) vs TPR = TP / (TP + FN) (y-axis)
    fpr = [0] + [fp[i] / (fp[i] + tn[i]) for i in range(len(fp))] + [1]
    tpr = [0] + [tp[i] / (tp[i] + fn[i]) for i in range(len(tp))] + [1]
    auc = sum([tpr[i] * (fpr[i] - fpr[i - 1]) for i in range(1, len(tpr))])

    assert auc >= 0 and auc <= 1

    for i in range(1, len(fpr)):
        assert fpr[i] >= fpr[i - 1]
        assert tpr[i] >= tpr[i - 1]

    # Welch's t-test
    welch_t, p_val = ttest_ind(_01_score, _02_score, equal_var=False)

    return acc, fpr, tpr, auc, threshold, welch_t, p_val

def calc_dof_welch(s1, n1, s2, n2):
    a = (s1 ** 2) / n1 
    b = (s2 ** 2) / n2
    numerator = (a + b) ** 2
    denominator = ( (a ** 2) / (n1 - 1) ) + ( (b ** 2) / (n2 - 1) )
    return numerator / denominator

def create_output_dir(output_root):
    from datetime import datetime
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(output_root, datetime_str)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_filename(img_filepath):
    if type(img_filepath) is tuple or type(img_filepath) is list:
        assert len(img_filepath) == 1
        img_filepath = img_filepath[0]
    return img_filepath.split('/')[-1]

def get_int_fgrp_from_filepath(img_filepath):
    return int(get_fgrp(get_filename(img_filepath)))

# Returns whether we are doing 1-1 matching
def is_one_to_one(num_anchors, num_pos, num_neg):
    return num_anchors == 1 and num_pos == 1 and num_neg == 1

"""
Returns: (_01_dist, _02_dist, f2f_dist)
-> _01_dist containing distances between anchor and positive examples (from test_dataloader, using embedder)
-> _02_dist containing distances between anchor and negative examples
-> matrix f2f_dist of finger by finger distances where f2f_dist[i][j][k] = 
    list of distances between FGRP (i, j) for matching status k (SAME_PERSON or DIFF_PERSON)
-> tp_names, fp_names, tn_names, fn_names (iff threshold is given)
"""
def run_test_loop(test_dataloader, num_anchors, num_pos, num_neg):

    data_iter = iter(test_dataloader)
    assert batch_size == 1

    posPairs = []
    # loop through all the data
    for i in range(len(test_dataloader)):
        # test_images is 3 (anchor, pos, neg) * N (number of sample images) * image_size (1*3*224*224)
        test_images, test_labels, test_filepaths = next(data_iter)
        assert len(test_images) == 3
        assert len(test_images[0]) == num_anchors and len(test_images[1]) == num_pos and len(test_images[2]) == num_neg

        # TEST CODE
        # print(i)
        posPairs.append(
            {
                "anchor": test_filepaths[0][0],
                "pos": test_filepaths[1][0]
            }
        )
        # for j, name in zip([0, 1, 2], ['anchor', 'pos', 'neg']):
        #     print('\t{}:'.format(name), [get_filename(file) for file in test_filepaths[j]])
        # TEST CODE
    
    with open("pos_pairs.json", 'w') as outFile:
        json.dump(posPairs, outFile, indent=4)

    return

"""
Saves a ROC curve
"""
def plot_roc_auc(fpr, tpr, dataset_name, num_anchors, num_pos, num_neg):
    plt.plot([100 * val for val in fpr], [100 * val for val in tpr], label='NBIS Exact Fingerprint Matcher')
    plt.plot([0.0, 100.0], [0.0, 100.0], label='Baseline: Same-Person Fingerprints Indistinguishable', linestyle='--')
    plt.xlabel('FPR: % of different-finger pairs misidentified as the same finger')
    plt.ylabel('TPR: % of same-finger pairs correctly matched')
    plt.title('ROC Curve: Exact Fingerprint Matching of Same-Person Pairs Only')
    plt.gca().set_xticks([i * 10 for i in range(11)])
    plt.gca().set_yticks([i * 10 for i in range(11)])
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve_{}_{}_{}_{}.pdf'.format(\
        dataset_name, num_anchors, num_pos, num_neg)))
    plt.savefig(os.path.join(output_dir, 'roc_curve_{}_{}_{}_{}.png'.format(\
        dataset_name, num_anchors, num_pos, num_neg)))
    plt.clf(); plt.close()

    return

"""
Returns:
-> FingerprintDataset from the_data_folder, guaranteed to be testing
-> MultipleFingerDataset from FingerprintDataset, which has triplets of anchor, pos, neg fingerprints;
    number according to scale_factor, size fo triplets according to (num_anchors, num_pos, num_neg)
-> DataLoader with batch size 1 for MultipleFingerDataset 
"""
def load_data(the_data_folder, num_anchors, num_pos, num_neg, scale_factor, \
        diff_fingers_across_sets=True, diff_fingers_within_set=True, \
        diff_sensors_across_sets=True, same_sensor_within_set=True, \
        possible_fgrps=ALL_FINGERS):
    assert set(possible_fgrps).issubset(ALL_FINGERS)
    fingerprint_dataset = FingerprintDataset(os.path.join(the_data_folder, 'test'), train=False)
    test_dataset = MultipleFingerDataset(fingerprint_dataset, num_anchors, num_pos, num_neg, \
        SCALE_FACTOR=scale_factor, \
        diff_fingers_across_sets=diff_fingers_across_sets, diff_fingers_within_set=diff_fingers_within_set, \
        diff_sensors_across_sets=diff_sensors_across_sets, same_sensor_within_set=same_sensor_within_set, \
        acceptable_anchor_fgrps=possible_fgrps, acceptable_pos_fgrps=possible_fgrps, acceptable_neg_fgrps=possible_fgrps)
    print('loaded test dataset: {}'.format(the_data_folder))
    assert batch_size == 1
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return fingerprint_dataset, test_dataset, test_dataloader

def pos_pairs_bozo_runner(output_dir, mindtct_root, bozo_path):
    with open("pos_pairs.json", 'r') as inFile:
        pairsData = json.load(inFile)

    pairsGroup = []
    numSameFinger = 0
    for i in range(len(pairsData)):
        pair = pairsData[i]
        anchorId = get_int_fgrp_from_filepath(pair["anchor"]) # int(pair["anchor"].split("_")[-1][:2])
        posId = get_int_fgrp_from_filepath(pair["pos"]) # int(pair["pos"].split("_")[-1][:2])
        pairsGroup.append(frozenset([pair["anchor"], pair["pos"]]))
        if anchorId == posId:
            numSameFinger += 1

    print(numSameFinger, " / ", len(pairsData), " = ", (numSameFinger / len(pairsData)) * 100, "%")

    print("length of starting group = ", len(pairsGroup))

    pairsGroup = set(pairsGroup)

    print("length of unique set = ", len(pairsGroup))

    pairsGroup = [tuple(pair) for pair in pairsGroup]

    dataPath = mindtct_root
    outFilePath = os.path.join(output_dir, "sd302_pos.txt")

    invalidFingerprints = []
    for pair in pairsGroup:
        anchorFilepath = pair[0]
        anchorPersonId = anchorFilepath.split('/')[-2]
        anchorFilename = anchorFilepath.split('/')[-1]
        posFilepath = pair[1]
        posPersonId = posFilepath.split('/')[-2]
        posFilename = posFilepath.split('/')[-1]

        anchorMinutaePath = os.path.join(dataPath, anchorPersonId, anchorFilename + ".xyt")
        posMinutaePath = os.path.join(dataPath, posPersonId, posFilename + ".xyt")
        command = "{} {} {}".format(bozo_path, anchorMinutaePath, posMinutaePath)
        # print(command)
        # assert os.path.exists(anchorMinutaePath)
        # assert os.path.exists(posMinutaePath)
        if not os.path.exists(anchorMinutaePath):
            invalidFingerprints.append(anchorMinutaePath)
            if not os.path.exists(posMinutaePath):
                invalidFingerprints.append(posMinutaePath)
            continue
        result = int(subprocess.check_output(command, shell=True))
        with open(outFilePath, 'a') as outFile:
            outFile.write(anchorFilepath + " " + posFilepath + " " + str(result) + "\n")
        print("Checked the pair ", pair)

    with open("invalid_fingerprints.txt", 'w') as outFile:
        outFile.write(str(invalidFingerprints))

def pos_pairs_analysis(dataset_name, num_anchors, num_pos, num_neg, output_dir):
    _01_score = []
    _02_score = []
    with open(os.path.join(output_dir, "sd302_pos.txt"), 'r') as inFile:
        lines = inFile.readlines()
    for line in lines:
        line = line.split()
        anchorId = get_int_fgrp_from_filepath(line[0])
        posId = get_int_fgrp_from_filepath(line[1])
        similarityScore = int(line[2])
        if anchorId == posId:
            _01_score.append(similarityScore)
        else:
            _02_score.append(similarityScore)

    # CALCULATE ACCURACY AND ROC AUC
    accs, fpr, tpr, auc, threshold, welch_t, p_val = get_metrics(_01_score, _02_score)

    plot_roc_auc(fpr=fpr, tpr=tpr, \
        dataset_name=dataset_name, \
        num_anchors=num_anchors, num_pos=num_pos, num_neg=num_neg)
    
    print("Num same fingers ", len(_01_score))
    print("Num different fingers ", len(_02_score))
    print("ROC AUC ", auc)
    


def main(the_data_folder, output_dir, num_anchors, bozo_path, mindtct_root, \
        num_pos, num_neg, scale_factor=1, \
        diff_fingers_across_sets=True, diff_fingers_within_set=True, \
        diff_sensors_across_sets=True, same_sensor_within_set=True, \
        possible_fgrps=' '.join(ALL_FINGERS)):
    print('Number anchor, pos, neg fingers: {}, {}, {}'.format(num_anchors, num_pos, num_neg))

    fingerprint_dataset, test_dataset, test_dataloader = \
        load_data(the_data_folder=the_data_folder, \
        num_anchors=num_anchors, num_pos=num_pos, num_neg=num_neg, \
        scale_factor=scale_factor, \
        diff_fingers_across_sets=diff_fingers_across_sets, diff_fingers_within_set=diff_fingers_within_set, \
        diff_sensors_across_sets=diff_sensors_across_sets, same_sensor_within_set=same_sensor_within_set, \
        possible_fgrps=possible_fgrps.split())

    dataset_name = the_data_folder[:-1 if the_data_folder[-1] == '/' else len(the_data_folder)].split('/')[-1]
    print('dataset name:', dataset_name)

    # TEST
    assert batch_size == 1

    run_test_loop(test_dataloader=test_dataloader, \
        num_anchors=num_anchors, num_pos=num_pos, num_neg=num_neg)
    
    pos_pairs_bozo_runner(output_dir, mindtct_root, bozo_path)
    
    pos_pairs_analysis(dataset_name=dataset_name, num_anchors=num_anchors, num_pos=num_pos,
                       num_neg=num_neg, output_dir=output_dir)
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser('gt_pairs_generator.py')
    parser.add_argument('--dataset', '-d', help='Path to folders containing images', type=str)
    parser.add_argument('--num_fingers', '-n', nargs='?', help='Number of fingers to test', \
        const=1, default=1, type=int)
    parser.add_argument('--output_root', '-o', nargs='?', help='Root directory for output', \
        const=DEFAULT_OUTPUT_ROOT, default=DEFAULT_OUTPUT_ROOT, type=str)
    parser.add_argument('--bozo_path', '-b', nargs='?', help='Path to bozo executable', \
        const="/home/aniv/NBIS_Installation/bin/bozorth3", default="/home/aniv/NBIS_Installation/bin/bozorth3", type=str)
    parser.add_argument('--mindtct_root', '-m', nargs='?', help='Path to root of mindtct outputs for sd302', \
        const="/data/verifiedanivray/mindtct_output/sd302_split/", default="/data/verifiedanivray/mindtct_output/sd302_split/", type=str)
    parser.add_argument('--scale_factor', '-s', nargs='?', help='Number of times to loop through the dataset to create triplets', \
        const=1, default=1, type=int)
    parser.add_argument('--diff_fingers_across_sets', '-dfs', \
        help='Force fingers in different sets to be from fingers (e.g., can\'t have pair of two right index fingers)', \
        action='store_true')
    parser.add_argument('--diff_fingers_within_set', '-dws', \
        help='Force fingers within set to be from distinct fingers', \
        action='store_true')
    parser.add_argument('--diff_sensors_across_sets', '-dss', \
        help='Force fingerprints in different sets to come from different sensors', \
        action='store_true')
    parser.add_argument('--same_sensor_within_set', '-sss', \
        help='Force all fingerprints in a set to come from the same sensor', \
        action='store_true')
    parser.add_argument('--possible_fgrps', type=str, default='01 02 03 04 05 06 07 08 09 10',
        help='Possible finger types to use in analysis (default: \'01 02 03 04 05 06 07 08 09 10\')')

    args = parser.parse_args()

    dataset = args.dataset
    num_fingers = args.num_fingers
    output_dir = create_output_dir(args.output_root)
    bozo_path = args.bozo_path
    mindtct_root = args.mindtct_root
    scale_factor = args.scale_factor

    diff_fingers_across_sets = args.diff_fingers_across_sets
    diff_fingers_within_set = args.diff_fingers_within_set
    diff_sensors_across_sets = args.diff_sensors_across_sets
    same_sensor_within_set = args.same_sensor_within_set

    possible_fgrps = args.possible_fgrps

    print(args)

    assert num_fingers > 0
    assert scale_factor >= 1

    main(the_data_folder=dataset, output_dir=output_dir, bozo_path=bozo_path, mindtct_root=mindtct_root, \
        num_anchors=num_fingers, num_pos=num_fingers, num_neg=num_fingers, \
        scale_factor=scale_factor, \
        diff_fingers_across_sets=diff_fingers_across_sets, diff_fingers_within_set=diff_fingers_within_set, \
        diff_sensors_across_sets=diff_sensors_across_sets, same_sensor_within_set=same_sensor_within_set, \
        possible_fgrps=possible_fgrps)

    # TESTED - pass!
    # Default command python pos_pairs_tester.py -d /data/therealgabeguo/fingerprint_data/sd302_split -o /data/verifiedanivray/results