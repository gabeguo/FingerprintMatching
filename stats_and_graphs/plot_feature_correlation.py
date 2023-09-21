import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../dl_models')
from parameterized_multiple_finger_tester import NUM_SAMPLES_KEY, NUM_POS_PAIRS_KEY, NUM_NEG_PAIRS_KEY, ROC_AUC_KEY, P_VAL_KEY
import json

ORIGINAL = 'Raw\nImage'
BINARIZED = 'Binarized\nImage'
RIDGE_ORIENT = 'Ridge\nOrientation\nMap'
RIDGE_FREQ = 'Ridge\nDensity\nMap'
MINUTIAE_LOC = 'Minutiae\nLocation'
MINUTIAE_FULL = 'Minutiae\nMap'

datasetNames = [ORIGINAL, BINARIZED, RIDGE_ORIENT, RIDGE_FREQ, MINUTIAE_FULL]
datasetName_2_resultPath = {
    ORIGINAL: '/data/therealgabeguo/updated_fingerprint_results_fall23/paper_results/feature_correlation/sd302/unpretrained_sd302/2023-09-20_23:21:33/test_results_sd302_split_feature_model_unpretrained_sd302_1_1_1.json',
    BINARIZED: '/data/therealgabeguo/updated_fingerprint_results_fall23/paper_results/feature_correlation/sd302/enhanced/2023-09-20_23:11:51/test_results_enhanced_feature_model_enhanced_1_1_1.json',
    RIDGE_ORIENT: '/data/therealgabeguo/updated_fingerprint_results_fall23/paper_results/feature_correlation/sd302/orient/2023-09-20_23:15:19/test_results_orient_feature_model_orient_1_1_1.json',
    RIDGE_FREQ: '/data/therealgabeguo/updated_fingerprint_results_fall23/paper_results/feature_correlation/sd302/freq/2023-09-20_23:18:31/test_results_freq_feature_model_freq_1_1_1.json',
    MINUTIAE_FULL: '/data/therealgabeguo/updated_fingerprint_results_fall23/paper_results/feature_correlation/sd302/minutiae/2023-09-20_23:24:43/test_results_sd302_feature_model_minutiae_1_1_1.json',
}
datasetName_2_representativeImage = {
    ORIGINAL: '/data/therealgabeguo/fingerprint_data/sd302_split/test/00002332/00002332_A_roll_05.png',
    BINARIZED: '/data/therealgabeguo/fingerprint_data/sd302_feature_extractions/enhanced/test/00002332/00002332_A_roll_05.png',
    RIDGE_ORIENT: '/data/therealgabeguo/fingerprint_data/sd302_feature_extractions/orient/test/00002332/00002332_A_roll_05.png',
    RIDGE_FREQ: '/data/therealgabeguo/fingerprint_data/sd302_feature_extractions/freq/test/00002332/00002332_A_roll_05.png',
    MINUTIAE_FULL: '/data/therealgabeguo/fingerprint_data/mindtct_minutiae/sd302/test/00002332/00002332_A_roll_05.png',
}

name_2_rocAuc = dict()
name_2_pVal = dict()
name_2_size = dict()
name_2_numSamples = dict()

patterns = ["x", "+", '.', '/', '\\']
colors = ['skyblue', 'lightcoral', 'palegreen', 'mediumorchid', 'gold']

alpha = 1e-4

for datasetName in datasetNames:
    resultPath = datasetName_2_resultPath[datasetName]
    with open(resultPath, 'r') as fin:
        the_results = json.load(fin)
        name_2_rocAuc[datasetName] = the_results[ROC_AUC_KEY]
        name_2_pVal[datasetName] = the_results[PAIRED_P_VAL_KEY]
        name_2_size[datasetName] = the_results[NUM_SAMPLES_KEY]
        name_2_numSamples[datasetName] = the_results[NUM_POS_PAIRS_KEY] + the_results[NUM_NEG_PAIRS_KEY]

plt.figure(dpi=900)

plt.xlim([0, 1])
plt.ylim([0.05, 0.85])
y_pos = [0.15, 0.30, 0.45, 0.60, 0.75]
plt.yticks(y_pos, datasetNames, fontsize=9)

plt.subplots_adjust(left=0.1)

for i in range(len(datasetNames)):
    the_name = datasetNames[i]
    plt.barh(y_pos[i], name_2_rocAuc[the_name], height=0.12, color = colors.pop(), hatch=patterns.pop(), \
        label='{}: {} pairs \n({} fingerprints)'.format(the_name.split('\n')[-2], \
        name_2_numSamples[the_name], name_2_size[the_name]))
    p_test_passed = 'p {} {:.0e}'.format('<' if name_2_pVal[the_name] < alpha else '>=', alpha)
    plt.text(name_2_rocAuc[the_name] * 1.01, y_pos[i] - 0.025, "{:.3f}\n({})".format(name_2_rocAuc[the_name], p_test_passed), \
        color='black', fontsize=8)
    img = plt.imread(datasetName_2_representativeImage[the_name])
    origin = 'lower' if the_name == MINUTIAE_FULL else 'upper'
    plt.imshow(img, extent=[0, 0.1, \
        y_pos[i] - 0.05, y_pos[i] + 0.05], aspect='equal', zorder=2, cmap='gray', origin=origin)
plt.xlabel('ROC AUC', fontsize=11)
plt.ylabel('Dataset', fontsize=11)

plt.title('ROC AUC by Feature Map')

ax = plt.subplot(111)
# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])

plt.axvline(x=0.5, color='k', linestyle='--', linewidth='1', zorder=-1, label='Baseline\nPerformance')

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3, fontsize=9)

plt.savefig('output/feature_correlation.pdf')
plt.savefig('output/feature_correlation.png')