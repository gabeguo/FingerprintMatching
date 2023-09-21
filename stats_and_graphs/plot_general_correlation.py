import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../dl_models')
from parameterized_multiple_finger_tester import NUM_SAMPLES_KEY, NUM_POS_PAIRS_KEY, NUM_NEG_PAIRS_KEY, ROC_AUC_KEY, P_VAL_KEY
import json

datasetName_2_resultPath = {
    'SD300': '/data/therealgabeguo/updated_fingerprint_results_fall23/paper_results/proving_correlation/general_PRETRAINED/sd300_full/2023-09-20_23:08:12/test_results_sd300a_split_full_based_model_PRETRAINED_1_1_1.json',
    'SD301\n(true\nholdout)': '/data/therealgabeguo/updated_fingerprint_results_fall23/paper_results/proving_correlation/general_PRETRAINED/sd301_full/2023-09-20_23:09:19/test_results_sd301_split_full_based_model_PRETRAINED_1_1_1.json',
    'SD302': '/data/therealgabeguo/updated_fingerprint_results_fall23/paper_results/proving_correlation/general_PRETRAINED/sd302_full/2023-09-20_23:13:08/test_results_sd302_split_full_based_model_PRETRAINED_1_1_1.json'
}

name_2_rocAuc = dict()
name_2_pVal = dict()
name_2_size = dict()
name_2_numSamples = dict()

patterns = ["x", "+", 'o']
colors = ['deepskyblue', 'salmon', 'gold', ]

alpha = 1e-4

for datasetName in datasetName_2_resultPath:
    resultPath = datasetName_2_resultPath[datasetName]
    with open(resultPath, 'r') as fin:
        the_results = json.load(fin)
        name_2_rocAuc[datasetName] = the_results[ROC_AUC_KEY]
        name_2_pVal[datasetName] = the_results[PAIRED_P_VAL_KEY]
        name_2_size[datasetName] = the_results[NUM_SAMPLES_KEY]
        name_2_numSamples[datasetName] = the_results[NUM_POS_PAIRS_KEY] + the_results[NUM_NEG_PAIRS_KEY]

labels = [name for name in datasetName_2_resultPath]

plt.xlim([0, 1])
plt.ylim([0.1, 0.9])
y_pos = [0.25, 0.5, 0.75]#[0.325, 0.675]
plt.yticks(y_pos, labels)

plt.subplots_adjust(left=0.15)

for i in range(len(labels)):
    the_name = labels[i]
    plt.barh(y_pos[i], name_2_rocAuc[the_name], height=0.2, color = colors.pop(), hatch=patterns.pop(), \
        label='{}: \n{} pairs \n({} prints)'.format(the_name.split('\n')[0], \
        name_2_numSamples[the_name], name_2_size[the_name]))
    p_test_passed = 'p {} {:.0e}'.format('<' if name_2_pVal[the_name] < alpha else '>=', alpha)
    plt.text(name_2_rocAuc[the_name] * 1.01, y_pos[i] - 0.025, "{:.3f}\n({})".format(name_2_rocAuc[the_name], p_test_passed), \
        color='black')
plt.xlabel('ROC AUC')
plt.ylabel('Dataset')

plt.title('ROC AUC by Dataset')

ax = plt.subplot(111)
# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.175,
                 box.width, box.height * 0.85])

plt.axvline(x=0.5, color='k', linestyle='--', linewidth='1', zorder=-1, label='Baseline')

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=4)

plt.savefig('output/general_correlation.pdf')
plt.savefig('output/general_correlation.png')