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
import math
import random
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
from statistical_analyses.bayes_analysis import *

from common_filepaths import *

# Default Values
DEFAULT_OUTPUT_ROOT = '/data/verifiedanivray/results'
DEFAULT_LEADS_FILE = 'geometric_analysis_results'
DEFAULT_CUDA = 'cuda:2'

# Batch Size constant
batch_size=32
assert batch_size > 1 # can't be 1, has to be more (otherwise, squeezing tensors bugs out)

# JSON Keys
DATASET_KEY = 'dataset'
WEIGHTS_KEY = 'weights'
CUDA_KEY = 'cuda'
OUTPUT_DIR_KEY = 'output dir'
NUM_ANCHOR_KEY = 'number anchor fingers per set'
NUM_POS_KEY = 'number positive fingers per set'
NUM_NEG_KEY = 'number negative fingers per set'
PRIOR_KEY = '% positive samples'
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
LEADS_FP_KEY = 'Leads analysis false positive rate'
LEADS_FN_KEY = 'Leads analysis false negative rate'
LEADS_RESULT_KEY = 'Leads analysis results'

# More constants
SAME_PERSON = 0
DIFF_PERSON = 1

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
    fnr = [1] + [fn[i] / (fn[i] + tp[i]) for i in range(len(fn))] + [0]
    auc = sum([tpr[i] * (fpr[i] - fpr[i - 1]) for i in range(1, len(tpr))])
    f1score = [0] + [tp[i] / (tp[i] + (fp[i] + fn[i])/2) for i in range(len(fn))] + [0]

    assert auc >= 0 and auc <= 1

    for i in range(1, len(fpr)):
        assert fpr[i] >= fpr[i - 1]
        assert tpr[i] >= tpr[i - 1]

    # Welch's t-test
    welch_t, p_val = ttest_ind(_01_dist, _02_dist, equal_var=False)

    return acc, fpr, tpr, fnr, auc, f1score, threshold, welch_t, p_val

"""
Input: 
-> matrix f2f_dist of finger by finger distances where f2f_dist[i][j][k] = 
    list of distances between FGRP (i, j) for matching status k (SAME_PERSON or DIFF_PERSON)
Returns:
-> Matrix of ROC AUC per finger pair
-> Matrix of p-value per finger pair
-> Matrix of number of samples per finger pair
-> Matrix of percent of samples that were same-person per finger pair
"""
def get_finger_by_finger_metrics(finger_to_finger_dist):
    finger_to_finger_num_samples = np.zeros((10, 10))
    finger_to_finger_percent_samePerson = np.zeros((10, 10))
    finger_to_finger_roc = np.zeros((10, 10))
    finger_to_finger_p_val = np.zeros((10, 10))
    for i in range(1, 10+1):
        for j in range(1, 10+1):
            same_person_dists = finger_to_finger_dist[i][j][SAME_PERSON]
            diff_person_dists = finger_to_finger_dist[i][j][DIFF_PERSON]

            _i, _j = i - 1, j - 1

            if len(same_person_dists) == 0 or len(diff_person_dists) == 0:
                finger_to_finger_roc[_i][_j] = np.NaN
                finger_to_finger_p_val[_i][_j] = np.NaN
                finger_to_finger_num_samples[_i][_j] = np.NaN
                finger_to_finger_percent_samePerson[_i][_j] = np.NaN
                continue

            accuracies, fpr, tpr, fnr, roc_auc, f1scores, threshold, welch_t, p_val = get_metrics(same_person_dists, diff_person_dists)
            
            finger_to_finger_roc[_i][_j] = roc_auc
            finger_to_finger_p_val[_i][_j] = p_val

            finger_to_finger_num_samples[_i][_j] = len(same_person_dists) + len(diff_person_dists)
            finger_to_finger_percent_samePerson[_i][_j] = len(same_person_dists) / finger_to_finger_num_samples[_i][_j]

    return finger_to_finger_roc, finger_to_finger_p_val, finger_to_finger_num_samples, finger_to_finger_percent_samePerson

def create_output_dir(output_root):
    from datetime import datetime
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(output_root, datetime_str)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_filename(img_filepath):
    # if type(img_filepath) is tuple or type(img_filepath) is list:
    #     assert len(img_filepath) == 1
    #     img_filepath = img_filepath[0]
    return img_filepath.split('/')[-1]

def get_int_fgrp_from_filepaths(img_filepath):
    return [int(get_fgrp(get_filename(x))) for x in img_filepath]

def get_pid_from_filepaths(img_filepath):
    return [get_id(get_filename(x)) for x in img_filepath]

def pids_match(filepaths_A, filepaths_B):
    return characteristics_match(get_pid_from_filepaths(filepaths_A), get_pid_from_filepaths(filepaths_B))

def fgrps_match(filepaths_A, filepaths_B):
    return characteristics_match(get_int_fgrp_from_filepaths(filepaths_A), get_int_fgrp_from_filepaths(filepaths_B))

def characteristics_match(characteristics_A, characteristics_B):
    #print(characteristics_A, '\n', characteristics_B)
    assert len(characteristics_A) == len(characteristics_B)

    num_matching = len([1 for i in range(len(characteristics_A)) if characteristics_A[i] == characteristics_B[i]])

    if num_matching == len(characteristics_A):
        return True
    elif num_matching == 0:
        return False
    raise ValueError('mixed matching and non-matching')

# Returns whether we are doing 1-1 matching
def is_one_to_one(num_anchors, num_pos, num_neg):
    return num_anchors == 1 and num_pos == 1 and num_neg == 1

"""
Returns: (_01_dist, _02_dist, f2f_dist)
-> _01_dist containing distances between anchor and positive examples (from test_dataloader, using embedder)
-> _02_dist containing distances between anchor and negative examples
-> matrix f2f_dist of finger by finger distances where f2f_dist[i][j][k] = 
    list of distances between FGRP (i, j) for matching status k (SAME_PERSON or DIFF_PERSON)
"""
def run_test_loop(test_dataloader, embedder, cuda, num_anchors, num_pos, num_neg, pos_indices):
    # distances between embedding of positive and negative pair
    _01_dist = []
    _02_dist = []
    # finger_to_finger_dist[i][j][k] = list of distances between FGRP (i, j) for matching status k (SAME_PERSON or DIFF_PERSON)
    finger_to_finger_dist = [[[[] for k in (SAME_PERSON, DIFF_PERSON)] for j in range(0, 10+1)] for i in range(0, 10+1)]

    data_iter = iter(test_dataloader)

    # tracks which pairs of fingerprint samples have been seen
    seen_pairs = set()

    # loop through all the data
    for i in range(len(test_dataloader)):
        isPos = i in pos_indices
        # test_images is 3 (anchor, pos, neg) * N (number of sample images) * image_size (1*3*224*224)
        test_images, test_labels, test_filepaths = next(data_iter)
        assert len(test_images) == 3
        #print(len(test_images[0]), len(test_images[1]), len(test_images[2]), num_anchors, num_pos, num_neg)
        assert len(test_images[0]) == num_anchors and len(test_images[1]) == num_pos and len(test_images[2]) == num_neg

        curr_anchor_pos_dists_by_batch = [[] for i in range(batch_size)]
        curr_anchor_neg_dists_by_batch = [[] for i in range(batch_size)]

        # batched pairwise comparison
        for i_a in range(num_anchors): # go through every anchor
            curr_anchor = test_images[0][i_a].to(cuda)
            embedding_anchor = torch.squeeze(embedder(curr_anchor))
            assert embedding_anchor.shape == (batch_size, 512) or i == len(test_dataloader) - 1 # one embedding for each item in batch

            anchor_filepaths = test_filepaths[0][i_a]
            anchor_fingers = get_int_fgrp_from_filepaths(anchor_filepaths)
            assert len(anchor_filepaths) == batch_size or i == len(test_dataloader) - 1
            assert len(anchor_fingers) == batch_size or i == len(test_dataloader) - 1

            for triplet_sameness_idx, sameness_code, num_samples, curr_dists_by_batch \
                    in zip([1, 2], [SAME_PERSON, DIFF_PERSON], [num_pos, num_neg], [curr_anchor_pos_dists_by_batch, curr_anchor_neg_dists_by_batch]):
                assert isPos == 1 or not isPos == 1 or isPos == 0 or not isPos == 0
                if (not isPos) != sameness_code: # skip the other pair, because we want to have a given proportion of positive samples
                    continue
                for i_curr in range(num_samples):
                    curr_sample = test_images[triplet_sameness_idx][i_curr].to(cuda)
                    embedding_curr = torch.squeeze(embedder(curr_sample))
                    assert embedding_curr.shape == (batch_size, 512) or i == len(test_dataloader) - 1

                    batched_dists = torch.square(nn.functional.pairwise_distance(embedding_anchor, embedding_curr)).tolist()
                    assert len(batched_dists) == batch_size or i == len(test_dataloader) - 1
                    for b in range(len(anchor_filepaths)): # can't average across different items in a batch, can only average embeddings for one item
                        curr_dists_by_batch[b].append(batched_dists[b])
                    
                    # finger-by-finger
                    curr_filepaths = test_filepaths[triplet_sameness_idx][i_curr]
                    curr_fingers = get_int_fgrp_from_filepaths(curr_filepaths)

                    assert len(curr_filepaths) == len(anchor_filepaths) and (len(curr_filepaths) == batch_size or i == len(test_dataloader) - 1)
                    assert len(curr_fingers) == len(anchor_fingers) and (len(curr_fingers) == batch_size or i == len(test_dataloader) - 1)
                    do_pids_match = pids_match(curr_filepaths, anchor_filepaths)
                    assert (isPos and do_pids_match) or ((not isPos) and (not do_pids_match))
                    assert not fgrps_match(curr_filepaths, anchor_filepaths)

                    # go through each item in the batch
                    for b in range(len(anchor_filepaths)):
                        anchor_filepath = anchor_filepaths[b]
                        curr_filepath = curr_filepaths[b]
                        anchor_finger = anchor_fingers[b]
                        curr_finger = curr_fingers[b]

                        if (anchor_filepath, curr_filepath) in seen_pairs or (curr_filepath, anchor_filepath) in seen_pairs:
                            continue # don't double-count for finger-by-finger
                        seen_pairs.add((anchor_filepath, curr_filepath))
                        seen_pairs.add((curr_filepath, anchor_filepath))
                        finger_to_finger_dist[anchor_finger][curr_finger][sameness_code].append(batched_dists[b])
                        if anchor_finger != curr_finger: # don't double-count same-finger pair
                            finger_to_finger_dist[curr_finger][anchor_finger][sameness_code].append(batched_dists[b])
        # process item-by-item from batch, and only average embeddings for that item (can't average everything within batch)
        for b in range(len(anchor_filepaths)):
            if isPos:
                _01_dist.append(np.mean(curr_anchor_pos_dists_by_batch[b]))
            else:
                _02_dist.append(np.mean(curr_anchor_neg_dists_by_batch[b]))

        if i % 5 == 0:
            print('Batch (item) {} out of {}'.format(i, len(test_dataloader)))
            print('\taverage, std squared L2 distance between positive pairs {:.3f}, {:.3f}'.format(np.mean(_01_dist), np.std(_01_dist)))
            print('\taverage, std squared L2 distance between negative pairs {:.3f}, {:.3f}'.format(np.mean(_02_dist), np.std(_02_dist)))
    
    return _01_dist, _02_dist, finger_to_finger_dist

"""
Saves a ROC curve
"""
def plot_roc_auc(fpr, tpr, dataset_name, weights_name, num_anchors, num_pos, num_neg):
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
        diff_sensors_across_sets=True, same_sensor_within_set=True):
    fingerprint_dataset = FingerprintDataset(os.path.join(the_data_folder, 'test'), train=False)
    test_dataset = MultipleFingerDataset(fingerprint_dataset, num_anchors, num_pos, num_neg, \
        SCALE_FACTOR=scale_factor, \
        diff_fingers_across_sets=diff_fingers_across_sets, diff_fingers_within_set=diff_fingers_within_set, \
        diff_sensors_across_sets=diff_sensors_across_sets, same_sensor_within_set=same_sensor_within_set)
    # TEST CODE TO SPEED UP
    # test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, len(test_dataset), 5)))
    # END TEST CODE
    print('loaded test dataset: {}'.format(the_data_folder))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return fingerprint_dataset, test_dataset, test_dataloader

def create_finger_by_finger_plot(f2f_data, the_title, the_cmap, the_fontsize, the_fmt):
    fgrp_names = ['Right Thumb', 'Right Index', 'Right Middle', 'Right Ring', 'Right Pinky', \
        'Left Thumb', 'Left Index', 'Left Middle', 'Left Ring', 'Left Pinky']
    plt.subplots_adjust(bottom=0.22, left=0.22)
    plt.title(the_title)
    sns.heatmap(f2f_data, annot=True, xticklabels=fgrp_names, yticklabels=fgrp_names, \
        cmap=the_cmap, fmt=the_fmt, annot_kws={"fontsize":the_fontsize})
    plt.savefig(os.path.join(output_dir, '{}.pdf'.format(the_title)))
    plt.savefig(os.path.join(output_dir, '{}.png'.format(the_title)))
    plt.clf(); plt.close()
    return

def calculate_baseline(prior, alpha):
    A = prior
    print("Probability of same person = ", A)
    i = 1
    while True:
        prob = geom.cdf(i, A)
        #if i % 10 == 0:
            #print("Probability with ", i, " samples = ", prob)
        if prob > alpha:
            print("It will take ", i - 1, "false positives to get the first true positive with ", alpha * 100, "% certainty")
            return i
        i += 1

def graph_geo_analysis(fpr, fnr, f1scores, alpha, num_fingers, leads_filename):
    output_root = args.output_root
    # Do the geometric distribution analysis
    # best accuracy heuristic accs.index(max(accs)) + 1
    best_index = f1scores.index(max(f1scores))
    print("'best' index: ", best_index, " with f1score: ", f1scores[best_index])
    # "best" false positive rate (as judged by the the best accuracy)
    fp = fpr[best_index]
    print("corresponding false positive rate: ", fp)
    # "best" false negative rate
    fn = fnr[best_index]
    print("corresponding false negative rate: ", fn)
    num_trials_needed = geometric_distribution(fp, fn, prior, alpha)

    # update results file
    with open(os.path.join(output_root, leads_filename), 'r') as inFile:
        data = json.load(inFile)
    priors = data["data"][str(num_fingers)]["prior"]
    num_fps = data["data"][str(num_fingers)]["num_fp"]
    if prior not in priors:
        priors.append(prior)
        num_fps.append(num_trials_needed)
    else:
        # overwrite it
        indexOfDatum = priors.index(prior)
        num_fps[indexOfDatum] = num_trials_needed
    points = [(priors[i], num_fps[i]) for i in range(len(priors))]
    points.sort(key=lambda x : x[0])
    data["data"][str(num_fingers)]["prior"] = [points[i][0] for i in range(len(points))]
    data["data"][str(num_fingers)]["num_fp"] = [points[i][1] for i in range(len(points))]
    with open(os.path.join(output_root, leads_filename), 'w') as outFile:
        json.dump(data, outFile, indent=4)
    
    # update graph
    markers = ['o', '*', '+', 'x', '^', 's']
    plt.yscale("log")
    plt.xscale("log")
    for finger_num in data["data"]:
        Xs = data["data"][finger_num]["prior"].copy()
        # Convert priors to dataset "size"
        #for i in range(len(Xs)):
        #    Xs[i] = 1.0 / Xs[i]
        Ys = data["data"][finger_num]["num_fp"]
        plt.plot(Xs, Ys, marker=markers.pop(), label="{} to {}".format(finger_num, finger_num))
    # calculate and plot the baseline
    Xs = data["data"][str(num_fingers)]["prior"].copy()
    baselines = []
    for i in range(len(Xs)):
        baselines.append(calculate_baseline(Xs[i], alpha))
        # Convert priors to dataset "size":
        # Xs[i] = 1.0 / Xs[i]
    plt.plot(Xs, baselines, marker=markers.pop(), label="exhaustive search")
    # # #
    # calculate and plot average
    Xs = data["data"]["1"]["prior"].copy()
    avgs = []
    for i in range(1, len(Xs)+1):
        #print("for ", Xs[-i])
        avg = 0
        for finger_num in data["data"]:
            #print(finger_num)
            if len(data["data"][finger_num]["num_fp"]) >= i:
                #print(data["data"][finger_num]["num_fp"][-i])
                avg += data["data"][finger_num]["num_fp"][-i]
        avg /= 5
        avgs.insert(0, avg)
    plt.plot(Xs, avgs, label="average", color="black", linewidth=3)
    # # #
    plt.xlabel('Proportion of Fingerprints Belonging to True Criminal')
    plt.ylabel('# of Leads Before Finding True Criminal')
    plt.title('Forensic Investigation Efficiency')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_root, 'geo_curve.pdf'))
    plt.savefig(os.path.join(output_root, 'geo_curve.png'))
    plt.clf(); plt.close()
    return fp, fn, num_trials_needed

def main(the_data_folder, weights_path, cuda, output_dir, num_anchors, num_pos, num_neg, \
        prior=0.5, scale_factor=1, confidence=0.95, leads_filename=DEFAULT_LEADS_FILE, \
        diff_fingers_across_sets=True, diff_fingers_within_set=True, \
        diff_sensors_across_sets=True, same_sensor_within_set=True):
    print('Number anchor, pos, neg fingers: {}, {}, {}'.format(num_anchors, num_pos, num_neg))

    fingerprint_dataset, test_dataset, test_dataloader = \
        load_data(the_data_folder=the_data_folder, \
        num_anchors=num_anchors, num_pos=num_pos, num_neg=num_neg, \
        scale_factor=scale_factor, \
        diff_fingers_across_sets=diff_fingers_across_sets, diff_fingers_within_set=diff_fingers_within_set, \
        diff_sensors_across_sets=diff_sensors_across_sets, same_sensor_within_set=same_sensor_within_set)
    dataset_name = the_data_folder[:-1 if the_data_folder[-1] == '/' else len(the_data_folder)].split('/')[-1]
    print('dataset name:', dataset_name)
    weights_name = (weights_path.split('/')[-1])[:-4]
    print('weights name:', weights_name)

    # CREATE EMBEDDER
    embedder = EmbeddingNet()

    # LOAD MODEL

    embedder.load_state_dict(torch.load(weights_path))
    embedder.eval()
    embedder.to(cuda)

    # TEST

    # Select Positives
    dataset_len = len(test_dataloader)
    all_indices = [i for i in range(dataset_len)]
    num_pos_samples = int(dataset_len * prior)
    pos_indices = random.sample(all_indices, num_pos_samples)
    pos_indices.sort()

    _01_dist, _02_dist, finger_to_finger_dist = run_test_loop(test_dataloader=test_dataloader, embedder=embedder, cuda=cuda, \
        num_anchors=num_anchors, num_pos=num_pos, num_neg=num_neg, pos_indices=pos_indices)

    f2f_roc, f2f_p_val, f2f_num_samples, f2f_percent_samePerson = get_finger_by_finger_metrics(finger_to_finger_dist)
    create_finger_by_finger_plot(f2f_roc, 'Finger-to-Finger ROC AUC', the_cmap='Reds', the_fontsize=9, the_fmt='.2f')
    create_finger_by_finger_plot(f2f_p_val, 'Finger-to-Finger P-Value (Welch\'s T-Test)', the_cmap='Purples', the_fontsize=6, the_fmt='.2g')
    create_finger_by_finger_plot(f2f_num_samples, 'Number of Finger-to-Finger Pairs', the_cmap='Blues', the_fontsize=9, the_fmt='g')
    create_finger_by_finger_plot(f2f_percent_samePerson, 'Proportion of Same-Person Samples', the_cmap='Greens', the_fontsize=9, the_fmt='.2f')

    # CALCULATE ACCURACY AND ROC AUC
    accs, fpr, tpr, fnr, auc, f1scores, threshold, welch_t, p_val = get_metrics(_01_dist, _02_dist)

    plot_roc_auc(fpr=fpr, tpr=tpr, \
        dataset_name=dataset_name, weights_name=weights_name, \
        num_anchors=num_anchors, num_pos=num_pos, num_neg=num_neg)

    fp, fn, num_false_leads = graph_geo_analysis(fpr, fnr, f1scores, confidence, num_anchors, leads_filename)

    # do the output
    final_results = {
        DATASET_KEY: the_data_folder, WEIGHTS_KEY: weights_path, CUDA_KEY: cuda,
        OUTPUT_DIR_KEY: output_dir,
        NUM_ANCHOR_KEY: num_anchors, NUM_POS_KEY: num_pos, NUM_NEG_KEY: num_neg,
        PRIOR_KEY: prior*100, SCALE_FACTOR_KEY: scale_factor,
        DIFF_FINGER_CROSS_SET_KEY: diff_fingers_across_sets, DIFF_FINGER_WITHIN_SET_KEY: diff_fingers_within_set,
        DIFF_SENSOR_CROSS_SET_KEY: diff_sensors_across_sets, SAME_SENSOR_WITHIN_SET_KEY: same_sensor_within_set,
        NUM_SAMPLES_KEY: len(fingerprint_dataset),
        NUM_POS_PAIRS_KEY: len(_01_dist), NUM_NEG_PAIRS_KEY: len(_02_dist), 
        MEAN_POS_DIST_KEY: np.mean(_01_dist), STD_POS_DIST_KEY: np.std(_01_dist),
        MEAN_NEG_DIST_KEY: np.mean(_02_dist), STD_NEG_DIST_KEY: np.std(_02_dist),
        ACC_KEY: max(accs), ROC_AUC_KEY: auc,
        T_VAL_KEY: welch_t, P_VAL_KEY: p_val,
        LEADS_FP_KEY: fp, LEADS_FN_KEY: fn, LEADS_RESULT_KEY: "It will take {} trials to get the first true positive with {}% certainty".format(num_false_leads, alpha*100)
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
    parser.add_argument('--prior', '-p', help='Proportion of positive samples in testing data', type=float)
    parser.add_argument('--scale_factor', '-s', nargs='?', help='Number of times to loop through the dataset to create triplets', \
        const=1, default=1, type=int)
    parser.add_argument('--alpha', '-a', help='Confidence used for leads analysis', type=float)
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
    parser.add_argument('--leads_output', '-l', nargs='?', help='Leads analysis data output filename', \
        const=DEFAULT_LEADS_FILE, default=DEFAULT_LEADS_FILE, type=str)

    args = parser.parse_args()

    dataset = args.dataset
    weights = args.weights
    cuda = args.cuda
    num_fingers = args.num_fingers
    output_dir = create_output_dir(args.output_root)
    prior = args.prior
    scale_factor = args.scale_factor
    alpha = args.alpha
    leads_data_filename = args.leads_output

    diff_fingers_across_sets = args.diff_fingers_across_sets
    diff_fingers_within_set = args.diff_fingers_within_set
    diff_sensors_across_sets = args.diff_sensors_across_sets
    same_sensor_within_set = args.same_sensor_within_set

    print(args)

    assert num_fingers > 0
    assert scale_factor >= 1

    main(the_data_folder=dataset, weights_path=weights, cuda=cuda, output_dir=output_dir, \
        num_anchors=num_fingers, num_pos=num_fingers, num_neg=num_fingers, \
        prior=prior, scale_factor=scale_factor, confidence=alpha, leads_filename=leads_data_filename, \
        diff_fingers_across_sets=diff_fingers_across_sets, diff_fingers_within_set=diff_fingers_within_set, \
        diff_sensors_across_sets=diff_sensors_across_sets, same_sensor_within_set=same_sensor_within_set)

    # TESTED - pass (distribution shift)