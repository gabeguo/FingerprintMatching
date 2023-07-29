import os
import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

RACIAL_GROUPS = ['white', 'non-white', 'race']
GENDER_GROUPS = ['female', 'male', 'gender']
POSSIBLE_GROUPINGS = {'Race': RACIAL_GROUPS, 'Gender': GENDER_GROUPS}

def get_group_name(filepath, is_weights_path):
    name = filepath.split('/')[-1]
    if is_weights_path:
        name = name[len('demographic_model_'):-4]
    return name

def capitalize(token):
    if 'sd' in token:
        return f'SD{token[2:]}'
    elif '-' in token:
        return '-'.join([subtoken.capitalize() for subtoken in token.split('-')])
    else:
        return token.capitalize()

# Get a list of all subdirectories
subdirs = glob.glob('/home/gabeguo/fingerprint_results/paper_results/fairness/sd302/*', recursive=False)

roc_auc_scores = {}

all_train_groups = set()
all_test_groups = set()

# Iterate through each subdirectory
for subdir in subdirs:
    #print(subdir)

    # Get a list of all JSON files in the subdirectory (and its subdirectories)
    json_files = glob.glob(os.path.join(subdir, '**/*.json'), recursive=True)

    scores = []
    weights = set()
    datasets = set()

    # Read each JSON file and extract the relevant information
    for json_file in json_files:
        #print(f'\t{json_file}')
        with open(json_file) as file:
            data = json.load(file)
            scores.append(data['ROC AUC'])
            weights.add(data['weights'])
            datasets.add(data['dataset'])

    # Check that all 'weights' and 'dataset' fields are the same in each directory
    if len(weights) > 1:
        print(f"Warning: Found multiple 'weights' values in {subdir}")
    if len(datasets) > 1:
        print(f"Warning: Found multiple 'dataset' values in {subdir}")

    train_group = get_group_name(list(weights)[0], is_weights_path=True)
    test_group = get_group_name(list(datasets)[0], is_weights_path=False)
    #if any([x in train_group for x in CURR_GROUPS]) and any([x in test_group for x in CURR_GROUPS]):
    all_train_groups.add(train_group)
    all_test_groups.add(test_group)

    # Compute mean and standard deviation of the ROC AUC scores
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)

    roc_auc_scores[(train_group, test_group)] = {'mean': mean_score, 'std': std_score}

for curr_grouping_name in POSSIBLE_GROUPINGS:
    curr_grouping = POSSIBLE_GROUPINGS[curr_grouping_name]
    curr_train_groups = [x for x in all_train_groups if any(y in x for y in curr_grouping)]
    curr_test_groups = [x for x in all_test_groups if any(y in x for y in curr_grouping)]
    curr_train_groups.sort()
    curr_test_groups.sort(); curr_test_groups.reverse()
    print(curr_train_groups, curr_test_groups)

    mean_roc_auc = np.zeros((len(curr_train_groups), len(curr_test_groups)))
    std_roc_auc = np.zeros((len(curr_train_groups), len(curr_test_groups)))

    #print('\n'.join([str(item) for item in roc_auc_scores]))

    for train_idx, train_group in enumerate(curr_train_groups):
        for test_idx, test_group in enumerate(curr_test_groups):
            mean_roc_auc[train_idx, test_idx] = roc_auc_scores[(train_group, test_group)]['mean']
            std_roc_auc[train_idx, test_idx] = roc_auc_scores[(train_group, test_group)]['std']

    # Create a heatmap with mean values, but annotations for both mean and std
    plt.figure(figsize=(8, 12))
    curr_test_groups = [' '.join([capitalize(token) for token in name.split('_')]) for name in curr_test_groups]
    curr_train_groups = [' '.join([capitalize(token) for token in name.split('_')]) for name in curr_train_groups]
    annot = [[f"{mean:.3f} ± {std:.3f}" for mean, std in zip(mean_row, std_row)] 
            for mean_row, std_row in zip(mean_roc_auc, std_roc_auc)]
    heatmap = sns.heatmap(mean_roc_auc, annot=annot, 
                fmt='', annot_kws={'fontsize':15}, cmap='coolwarm',
                xticklabels=curr_test_groups, yticklabels=curr_train_groups)
    heatmap.set_xlabel('Testing Group', fontsize=14)
    heatmap.set_ylabel('Training Group', fontsize=14)
    plt.title(f'ROC AUC (Mean ± Std Err): {curr_grouping_name} Demographics')
    plt.show()
