import os
import json
import sys; sys.path.append('../dl_models')
from parameterized_multiple_finger_tester import *

DATA_FOLDER = '/data/therealgabeguo/paper_results/multi_finger'
SD301_KEY = 'SD 301'
SD302_KEY = 'SD 302'

dataset_to_roc = {SD301_KEY: list(), SD302_KEY: list()}
dataset_to_num_samples = {SD301_KEY: list(), SD302_KEY: list()}
dataset_to_num_tuples = {SD301_KEY: list(), SD302_KEY: list()}

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

            dataset_key = SD301_KEY if 'sd301' in the_filepath.lower() else SD302_KEY
            dataset_to_roc[dataset_key].append((curr_n, curr_roc))

            dataset_to_num_samples[dataset_key].append(curr_num_samples)
            dataset_to_num_tuples[dataset_key].append(curr_num_tuples)

#print(dataset_to_roc)

import matplotlib.pyplot as plt

line_markers = ['o', 'X']
print(dataset_to_num_samples)
for key in dataset_to_roc:
    curr_data = dataset_to_roc[key]
    curr_num_distinct_fingerprints = set(dataset_to_num_samples[key])
    assert len(curr_num_distinct_fingerprints) == 1
    (curr_num_distinct_fingerprints,) = curr_num_distinct_fingerprints
    #print(curr_num_distinct_fingerprints)
    curr_num_tuples = set(dataset_to_num_tuples[key])
    assert len(curr_num_tuples) == 1
    (curr_num_tuples,) = curr_num_tuples
    #print(curr_num_tuples)

    curr_x = [item[0] for item in curr_data]
    curr_y = [item[1] for item in curr_data]
    plt.plot(curr_x, curr_y, line_markers.pop(0) + '-', \
        label='{} ({} samples, {} combos)'.format(key, curr_num_distinct_fingerprints, curr_num_tuples))
    for num_fingers, roc_auc in curr_data:
        plt.text(num_fingers - 0.15, roc_auc + 0.005, str(round(roc_auc, 3)))

plt.legend()
plt.xlim(0.75, 5.25)
plt.xticks([i for i in range(1, 6)], ['1-to-1', '2-to-2', '3-to-3', '4-to-4', '5-to-5'])
plt.xlabel('Number of Fingers')
plt.ylim(0.75, 0.95)
plt.ylabel('ROC AUC')
plt.title('N-to-N Disjoint Finger Matching Results')
plt.grid()

plt.savefig(os.path.join('output', 'multi_finger_results.pdf'))
plt.savefig(os.path.join('output', 'multi_finger_results.png'))