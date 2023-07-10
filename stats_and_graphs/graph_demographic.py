# Thanks ChatGPT!

import json
import matplotlib.pyplot as plt
import os

import argparse

# Initialize the argument parser
parser = argparse.ArgumentParser()

# Add argument
parser.add_argument('--gender', action='store_true', help='Whether to split by gender.')

# Parse the arguments
args = parser.parse_args()

# Paths to your JSON files
if args.gender:
    json_files = [
        "/data/therealgabeguo/demographic_fingerprint/paper_results/fairness/sd302/train_sd302_female_split_test_sd302_female_split/2023-07-10_13:45:47/test_results_sd302_female_split_demographic_model_sd302_female_split_1_1_1.json",
        "/data/therealgabeguo/demographic_fingerprint/paper_results/fairness/sd302/train_sd302_female_split_test_sd302_male_split/2023-07-10_13:37:56/test_results_sd302_male_split_demographic_model_sd302_female_split_1_1_1.json",
        "/data/therealgabeguo/demographic_fingerprint/paper_results/fairness/sd302/train_sd302_male_split_test_sd302_male_split/2023-07-10_14:00:17/test_results_sd302_male_split_demographic_model_sd302_male_split_1_1_1.json",
        "/data/therealgabeguo/demographic_fingerprint/paper_results/fairness/sd302/train_sd302_male_split_test_sd302_female_split/2023-07-10_14:03:03/test_results_sd302_female_split_demographic_model_sd302_male_split_1_1_1.json",
    ]
else:
    json_files = [
        "/data/therealgabeguo/demographic_fingerprint/paper_results/fairness/sd302/train_sd302_white_split_test_sd302_white_split/2023-07-10_13:51:46/test_results_sd302_white_split_demographic_model_sd302_white_split_1_1_1.json",
        "/data/therealgabeguo/demographic_fingerprint/paper_results/fairness/sd302/train_sd302_white_split_test_sd302_non-white_split/2023-07-10_13:54:41/test_results_sd302_non-white_split_demographic_model_sd302_white_split_1_1_1.json",
        "/data/therealgabeguo/demographic_fingerprint/paper_results/fairness/sd302/train_sd302_non-white_split_test_sd302_non-white_split/2023-07-10_13:58:30/test_results_sd302_non-white_split_demographic_model_sd302_non-white_split_1_1_1.json",
        "/data/therealgabeguo/demographic_fingerprint/paper_results/fairness/sd302/train_sd302_non-white_split_test_sd302_white_split/2023-07-10_13:56:31/test_results_sd302_white_split_demographic_model_sd302_non-white_split_1_1_1.json"
    ]

plt.figure(figsize=(10, 6))

# Initiate lists to store names and values
bar_names = []
bar_labels = []
roc_aucs = []
p_vals = []

# Colors and hatches
colors = ['#ff5555', '#dd7777', '#5555ff', '#7777dd']
hatches = ['/', '\\', '/', '\\']

# Get dataset stats
num_samples = dict()
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Extract gender from dataset and weights names
    dataset_gender = os.path.basename(data['dataset']).split('_')[-2]
    # Count number of folders in train, val, and test subdirectories
    for sub_dir in ["train", "val", "test"]:
        path = os.path.join(data["dataset"], sub_dir)
        try:
            folder_count = len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])
            #print(f"Number of folders in {path}: {folder_count}")
            num_samples[(dataset_gender, sub_dir)] = folder_count
        except FileNotFoundError:
            print(f"{path} does not exist")

# Make bar plots
for idx, json_file in enumerate(json_files):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract gender from dataset and weights names
    weights_gender = os.path.basename(data['weights']).split('_')[-2] # training
    dataset_gender = os.path.basename(data['dataset']).split('_')[-2] # testing

    # Create the bar name with additional information
    bar_label = f'Train: {weights_gender}\n({num_samples[(weights_gender, "train")]}+{num_samples[(weights_gender, "val")]} people),\n' \
                f'Test: {dataset_gender}\n({num_samples[(dataset_gender, "test")]} people)' #\
                # f'{data["number of distinct samples in dataset"]} samples,\n' \
                # f'{data["number of positive pairs"] + data["number of negative pairs"]} pairs)'
    bar_labels.append(bar_label)
    bar_name = f'{weights_gender} train,\n{dataset_gender} test'
    bar_names.append(bar_name)

    # Get ROC AUC value
    roc_aucs.append(data['ROC AUC'])
    p_vals.append(data['p-value'])

#print(bar_names)

# Now, plot the bar graph with different colors and hatches
bars = plt.barh(bar_names, roc_aucs, color=colors)
for bar, hatch, label in zip(bars, hatches, bar_labels):
    bar.set_hatch(hatch)
    bar.set_label(label)
plt.legend(bars, bar_labels, bbox_to_anchor=(1, 1), loc='upper right', fontsize=10)
#plt.xlabel('Group')
#plt.ylabel('ROC AUC')
plt.title(f'{"Gender" if args.gender else "Racial"} Disparity Comparison')

#plt.ylim(0, 1)

# Add values above the bars
for bar, curr_p_val in zip(bars, p_vals):
    xval = bar.get_width()
    plt.text(xval, bar.get_y() + bar.get_height() / 2, f"{round(xval, 3)}\n{'p < 0.01' if curr_p_val < 0.01 else 'p >= 0.01'}", ha='left', va='center')
plt.xlim(0, 1)

#plt.show()
plt.savefig(f'{"gender" if args.gender else "race"} experiments.png')