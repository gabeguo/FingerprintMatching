import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.append('../dl_models')
from parameterized_multiple_finger_tester import MEAN_TRIPLET_DIST_KEY, STD_TRIPLET_DIST_KEY, \
    NUM_SAMPLES_KEY, NUM_POS_PAIRS_KEY, NUM_NEG_PAIRS_KEY, DATASET_KEY

alt_names = {
    'SD300A':'SD300',
    'enhanced':'Binarized',
    'freq':'Ridge Frequency',
    'minutiae':'Minutiae',
    'orient':'Orientation'
}

# Thanks ChatGPT!
def plot_confidence_intervals(dataset_directory, savename):
    # Initialize empty lists to store data
    labels = []
    means = []
    errors = []

    # Loop through files in the directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(dataset_directory):
        for filename in filenames:
            if '.json' in filename:
                filepath = os.path.join(dirpath, filename)
                if '2023-09-20_' not in filepath:
                    print('exclude', filepath)
                    continue
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract mean, std, and sample size
                mean = data[MEAN_TRIPLET_DIST_KEY]
                std = data[STD_TRIPLET_DIST_KEY]
                assert data[NUM_POS_PAIRS_KEY] == data[NUM_NEG_PAIRS_KEY]
                sample_size = data[NUM_POS_PAIRS_KEY] #data[NUM_SAMPLES_KEY]

                # rescale std to be sample estimate of std, rather than population std
                std = std * np.sqrt(sample_size / (sample_size - 1))

                # Calculate 95% confidence intervals
                z_value = 1.96  # For 95% confidence
                ci = z_value * (std / np.sqrt(sample_size))

                # Extract the SD label from the filename
                sd_label = filename.split('_')[2]
                # do some renaming
                if 'feature_correlation' in dirpath and 'sd' in sd_label:
                    if 'minutiae' in filename:
                        sd_label = 'Minutiae'
                    else:
                        sd_label = 'Raw'
                if 'sd' in sd_label:
                    sd_label = sd_label.upper()
                if sd_label in alt_names:
                    sd_label = alt_names[sd_label]

                # Append to lists
                labels.append(sd_label)
                means.append(mean)
                errors.append(ci)

    # Sort the lists by labels for better presentation
    sorted_indices = np.argsort(labels)
    labels = np.array(labels)[sorted_indices]
    means = np.array(means)[sorted_indices]
    errors = np.array(errors)[sorted_indices]

    print(labels)

    # Create plot
    fig, ax = plt.subplots()
    plt.errorbar(labels, means, marker='.', markersize=15, color='blue',
                 yerr=errors, alpha=0.9, ecolor='black', capsize=15, linestyle='none')
    for i in range(len(means)):
        plt.text(i, means[i] + 0.03, r'${} \pm {}$'.format(round(means[i], 3), round(errors[i], 3)), 
                 horizontalalignment='center')
    plt.ylabel(r'$d_{diff \ person} - d_{same \ person}$', fontsize=14)
    plt.xlabel('Dataset')
    #plt.ylabel(r'$\bar{x} \pm 1.96 \times \frac{s}{\sqrt{n}}$')
    plt.title('Effect Size of Cross-Finger Similarity:\n95% Confidence Interval')
    plt.xticks(labels)

    # Illustrate what is good and bad
    plt.axhline(y = 0, color = 'tab:brown', linestyle = '--')

    # set limits
    y_max = max(means[i] + errors[i] for i in range(len(means)))
    y_min = -y_max
    y_tick_min = (int(y_min / 0.1) - 3) * 0.1
    y_tick_max = (int(y_max / 0.1) + 3) * 0.1
    plt.yticks(np.arange(y_tick_min, y_tick_max, 0.1))
    plt.ylim(y_tick_min, y_tick_max)

    # Stylize
    plt.grid()

    # Show what is good and bad region
    x_min, x_max = ax.get_xlim()
    x_min -= 0.3
    x_max += 0.3
    x = np.arange(x_min, x_max + 0.1, 0.1)
    ax.fill_between(x, 0, y_tick_max, color=(0.9, 1, 0.9), alpha=0.6, label='Same Person Pairs More Similar')
    ax.fill_between(x, y_tick_min, 0, color=(1, 0.9, 0.9), alpha=0.6, label='Different Person Pairs More Similar')
    plt.xlim(x_min, x_max)
    plt.legend(labelcolor=[(0.1, 0.4, 0.1), (0.45, 0.15, 0.15)], framealpha=0.95)
    leg = ax.get_legend()
    leg.legend_handles[0].set_color('green')
    leg.legend_handles[1].set_color('red')

    # plt.text((x_max + x_min) / 2, y_tick_max * 0.25, r'Same Person Pairs More Similar', 
    #          color=(0.1, 0.3, 0.1), fontsize=12.5, fontweight='bold', horizontalalignment='center')
    # plt.text((x_max + x_min) / 2, y_tick_min * 0.25, r'Different Person Pairs More Similar', 
    #          color=(0.3, 0.1, 0.1), fontsize=12.5, fontweight='bold', horizontalalignment='center')

    # Display plot
    plt.savefig(f'{savename}.png')
    plt.savefig(f'{savename}.pdf')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot 95% confidence intervals based on .json files in a specified directory.')
    parser.add_argument('--dataset_directory', type=str, help='The directory containing the .json files',
                        default='/data/therealgabeguo/updated_fingerprint_results_fall23/paper_results/proving_correlation/general_PRETRAINED')
    parser.add_argument('--save_name', type=str, default='dummy')

    args = parser.parse_args()
    plot_confidence_intervals(args.dataset_directory, args.save_name)
