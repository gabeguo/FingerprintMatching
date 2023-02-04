import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import sys
import os
import math
import argparse
import random
import matplotlib.pyplot as plt
import json

sys.path.append('/home/aniv/FingerprintMatching/')
sys.path.append('/home/aniv/FingerprintMatching/directory_organization')

from trainer import *
from siamese_datasets import *
from fingerprint_dataset import *
from embedding_models import *
from fileProcessingUtil import *
from statistical_analyses.bayes_analysis import *

from common_filepaths import DATA_FOLDER, SUBSET_DATA_FOLDER, BALANCED_DATA_FOLDER, UNSEEN_DATA_FOLDER, EXTRA_DATA_FOLDER, \
    ENHANCED_DATA_FOLDER, ENHANCED_HOLDOUT_FOLDER, ENHANCED_INK_FOLDER, SYNTHETIC_DATA_FOLDER

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
    fnr = [1] + [fn[i] / (fn[i] + tp[i]) for i in range(len(fn))] + [0]
    auc = sum([tpr[i] * (fpr[i] - fpr[i - 1]) for i in range(1, len(tpr))])

    for i in range(1, len(fpr)):
        assert fpr[i] >= fpr[i - 1]
        assert tpr[i] >= tpr[i - 1]

    return acc, fpr, tpr, fnr, auc, threshold

def graph_geo_analysis():
    output_dir = '/data/verifiedanivray/results'
    with open(os.path.join(output_dir, 'geometric_analysis_results.json'), 'r') as inFile:
        data = json.load(inFile)
    Xs = data["data"]["prior"]
    Ys = data["data"]["num_fp"]
    plt.yscale("log")
    plt.xscale("log")
    plt.plot(Xs, Ys, marker='o')
    plt.xlabel('Prior of Positive Samples')
    plt.ylabel('# of False Positives Before True Positive')
    plt.title('Lead Efficiency Analysis')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'geo_curve.pdf'))
    plt.savefig(os.path.join(output_dir, 'geo_curve.png'))
    plt.clf(); plt.close()

def main(the_data_folder, MODEL_PATH, prior, cuda):
    print('weights used: {}\n\n'.format(MODEL_PATH))
    
    # Create output directory
    from datetime import datetime
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join('/data/verifiedanivray/results', datetime_str)
    os.makedirs(output_dir, exist_ok=True)

    # Data loading 
    batch_size=1
    test_dataset = TripletDataset(FingerprintDataset(os.path.join(the_data_folder, 'test'), train=False))
    print('loaded test dataset: {}'.format(the_data_folder))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # CREATE EMBEDDER
    embedder = EmbeddingNet()

    # CREATE TRIPLET NET
    siamese_net = SiameseNet(embedder)

    # distances between embedding of positive and negative pair
    _01_dist = []
    _02_dist = []

    # LOAD MODEL
    embedder.load_state_dict(torch.load(MODEL_PATH))
    embedder.eval()
    embedder = embedder.to(cuda)

    # SELECT POSITIVES
    dataset_len = len(test_dataloader)
    all_indices = [i for i in range(dataset_len)]
    num_pos = int(dataset_len * prior)
    pos_indices = random.sample(all_indices, num_pos)
    pos_indices.sort()
    print("dataset length ", dataset_len)
    print("num positives ", num_pos)
    # print("positive indices ", pos_indices)

    # TEST
    data_iter = iter(test_dataloader)
    for i in range(dataset_len):
        isPos = i in pos_indices
        test_images, test_labels, test_filepaths = next(data_iter)

        test_images = [item.to(cuda) for item in test_images]

        test_images = test_images[0:2] if isPos else test_images[1:3]

        #print(triplet_net(*test_images)[0].size())
        embeddings = [torch.reshape(e, (e.size()[0], e.size()[1])) for e in siamese_net(*test_images)]
        # len(embeddings) == 3 reprenting the following (anchor, pos, neg)
        # Each index in the list contains a tensor of size (batch size, embedding length)

        assert len(embeddings) == 2
        assert embeddings[0].size()[0] <= batch_size
        for batch_index in range(embeddings[0].size()[0]): # should be equivalent to range(batch_size):
            if isPos:
                _01_dist.append(euclideanDist(embeddings[0][batch_index], embeddings[1][batch_index]).item())
            else:
                _02_dist.append(euclideanDist(embeddings[0][batch_index], embeddings[1][batch_index]).item())

        if i % 40 == 0:
            print('Batch {} out of {}'.format(i, len(test_dataloader)))
            print('\taverage squared L2 distance between positive pairs:', np.mean(np.array(_01_dist)))
            print('\taverage squared L2 distance between negative pairs:', np.mean(np.array(_02_dist)))

    # CALCULATE ACCURACY AND ROC AUC
    accs, fpr, tpr, fnr, auc, threshold = get_metrics(_01_dist, _02_dist)

    #print("Accuracies\n", accs)
    #print("False Positives\n", fpr)
    #print("False Negatives\n", fnr)

    max_acc = max(accs)
    print('best accuracy:', max_acc)

    print('auc = {}'.format(auc))
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

    # Do the geometric distribution analysis
    # best accuracy heuristic accs.index(max(accs)) + 1
    distances = [ abs(tpr[i] - fpr[i]) for i in range(len(tpr)) ]
    best_index = distances.index(max(distances))
    print("Accuracy at 'best' index ", accs[best_index - 1])
    # "best" false positive rate (as judged by the the best accuracy)
    fp = fpr[best_index]
    print("corresponding false positive rate: ", fp)
    # "best" false negative rate
    fn = fnr[best_index]
    print("corresponding false negative rate: ", fn)
    # confidence level
    alpha = 0.95
    num_fps_needed = geometric_distribution(fp, fn, prior, alpha) - 1

    # update results file
    with open("/data/verifiedanivray/results/geometric_analysis_results.json", 'r') as inFile:
        data = json.load(inFile)
    priors = data["data"]["prior"]
    num_fps = data["data"]["num_fp"]
    if prior not in priors:
        priors.append(prior)
        num_fps.append(num_fps_needed)
    else:
        # overwrite it
        indexOfDatum = priors.index(prior)
        num_fps[indexOfDatum] = num_fps_needed
    points = [(priors[i], num_fps[i]) for i in range(len(priors))]
    points.sort(key=lambda x : x[0])
    data["data"]["prior"] = [points[i][0] for i in range(len(points))]
    data["data"]["num_fp"] = [points[i][1] for i in range(len(points))]
    with open("/data/verifiedanivray/results/geometric_analysis_results.json", 'w') as outFile:
        json.dump(data, outFile)

    # do the output
    
    results_fname = os.path.join(output_dir, 'test_results.txt')
    with open(results_fname, 'w') as fout:
        fout.write('weights used: {}\n\n'.format(MODEL_PATH))
        fout.write('data folder: {}\n\n'.format(the_data_folder))
        fout.write('{}% positive samples\n\n'.format(prior * 100))
        fout.write('average squared L2 distance between positive pairs: {}\n'.format(np.mean(_01_dist)))
        fout.write('std of  squared L2 distance between positive pairs: {}\n'.format(np.std(_01_dist)))
        fout.write('average squared L2 distance between negative pairs: {}\n'.format(np.mean(_02_dist)))
        fout.write('std of  squared L2 distance between negative pairs: {}\n'.format(np.std(_02_dist)))
        fout.write('best accuracy: {}\n'.format(str(max_acc)))
        fout.write('chosen false positive rate: {}\n'.format(str(fp)))
        fout.write('chosen false negative rate: {}\n'.format(str(fn)))
        fout.write('ROC AUC: {}\n\n'.format(auc))
        fout.write("It will take {} false positives to get the first true positive with {}% certainty".format(num_fps_needed, alpha*100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('parameterized_runner.py')
    parser.add_argument('--dataset', '-d', help='Path to folders containing images', type=str)
    parser.add_argument('--weights', '-w', help='Path to model weights', type=str)
    parser.add_argument('--prior', '-p', help='Proportion of positive samples in testing data', type=float)
    parser.add_argument('--cuda', '-c', help='Name of GPU we want to use', type=str)
    args = parser.parse_args()
    dataset = args.dataset
    weights = args.weights
    prior = args.prior
    cuda = args.cuda

    #main(dataset, weights, prior, cuda)
    graph_geo_analysis()