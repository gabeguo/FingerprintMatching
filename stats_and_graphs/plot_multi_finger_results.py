import os
import json
import sys; sys.path.append('../dl_models')
import numpy as np
from scipy.stats import sem
from parameterized_multiple_finger_tester import *

DATA_FOLDER = '/data/therealgabeguo/paper_results/multi_finger'
SD300_KEY = 'SD 300'
SD301_KEY = 'SD 301'
SD302_KEY = 'SD 302'

dataset_to_roc = {SD300_KEY: dict(), SD301_KEY: dict(), SD302_KEY: dict()}
dataset_to_num_samples = {SD300_KEY: set(), SD301_KEY: set(), SD302_KEY: set()}
dataset_to_num_tuples = {SD300_KEY: set(), SD301_KEY: set(), SD302_KEY: set()}

# find all the files and create ROC dict of lists
for root, dirs, files in os.walk(DATA_FOLDER, topdown=False):
    for the_file in files:
        if '.json' not in the_file:
            continue
        the_filepath = os.path.join(root, the_file)
        with open(the_filepath, 'r') as fin:
            the_info = json.load(fin)
            #print(json.dumps(the_info, indent=4))
            
            curr_roc = the_info[ROC_AUC_KEY]
            curr_num_samples = the_info[NUM_SAMPLES_KEY]
            curr_n = the_info[NUM_ANCHOR_KEY]
            curr_num_tuples = the_info[NUM_POS_PAIRS_KEY] + the_info[NUM_NEG_PAIRS_KEY] + \
                the_info[NUM_POS_PAIRS_KEY] # anchor combo (plus pos & neg)
            assert the_info[NUM_POS_PAIRS_KEY] == the_info[NUM_NEG_PAIRS_KEY]

            if 'sd300' in the_filepath.lower():
                dataset_key = SD300_KEY
            elif 'sd301' in the_filepath.lower():
                dataset_key = SD301_KEY
            elif 'sd302' in the_filepath.lower():
                dataset_key = SD302_KEY
            else:
                print('error - invalid filepath')
            if curr_n not in dataset_to_roc[dataset_key]:
                dataset_to_roc[dataset_key][curr_n] = list()
            dataset_to_roc[dataset_key][curr_n].append(curr_roc)

            dataset_to_num_samples[dataset_key].add(curr_num_samples)
            dataset_to_num_tuples[dataset_key].add(curr_num_tuples)

#print(dataset_to_roc)

import matplotlib.pyplot as plt

line_markers = ['x', '+', 'o']
print(dataset_to_num_samples)

n_trials = set([len(dataset_to_roc[the_dataset][n_fingers]) \
    for the_dataset in dataset_to_roc for n_fingers in dataset_to_roc[the_dataset]])
assert len(n_trials) == 1
n_trials = list(n_trials)[0]

colors = ['deepskyblue', 'salmon', 'goldenrod']

already_written_roc = set()

for key in dataset_to_roc:
    curr_data = dataset_to_roc[key]

    curr_num_distinct_fingerprints = dataset_to_num_samples[key]
    assert len(curr_num_distinct_fingerprints) == 1
    (curr_num_distinct_fingerprints,) = curr_num_distinct_fingerprints
    #print(curr_num_distinct_fingerprints)
    curr_num_tuples = dataset_to_num_tuples[key]
    assert len(curr_num_tuples) == 1
    (curr_num_tuples,) = curr_num_tuples
    #print(curr_num_tuples)

    curr_x = [n_fingers for n_fingers in curr_data]
    curr_y = [np.mean(curr_data[n_fingers]) for n_fingers in curr_data]
    yerr = [sem(curr_data[n_fingers]) for n_fingers in curr_data]
    print(curr_data)
    plt.errorbar(curr_x, curr_y, yerr=yerr, fmt=line_markers.pop(0) + '-', capsize=5, \
        label='{} ({} samples, {} combos)'.format(key, curr_num_distinct_fingerprints, curr_num_tuples), \
        color = colors.pop())
    for num_fingers in curr_data:
        roc_auc = np.mean(curr_data[num_fingers])
        if int(100 * round(roc_auc, 2)) in already_written_roc:
            plt.text(num_fingers - 0.15, roc_auc + 0.015, str(round(roc_auc, 3)))
        else:
            plt.text(num_fingers - 0.15, roc_auc + 0.005, str(round(roc_auc, 3)))

        already_written_roc.add(int(100 * round(roc_auc, 2)))

plt.legend()
plt.xlim(0.75, 5.25)
plt.xticks([i for i in range(1, 6)], ['1-to-1', '2-to-2', '3-to-3', '4-to-4', '5-to-5'])
plt.xlabel('Number of Fingers')
plt.ylim(0.70, 0.95)
plt.ylabel('ROC AUC ({} trials)'.format(n_trials))
plt.title('N-to-N Disjoint Finger Matching Results')
plt.grid()

plt.savefig(os.path.join('output', 'multi_finger_results.pdf'))
plt.savefig(os.path.join('output', 'multi_finger_results.png'))