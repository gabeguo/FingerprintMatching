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
from scipy.stats import sem

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

DEFAULT_OUTPUT_ROOT = '/data/verifiedanivray/results'

# pre: assuming all datafiles are complete and in the proper format
def percent_reduction_grapher(datafiles):
    # percent reduction in leads required
    output_root = args.output_root
    data_runs = []
    # calculate the average data
    for datafile in datafiles:
        with open(os.path.join(output_root, datafile), "r") as inFile:
            data_runs.append(json.load(inFile))

    # Collapse finger numbers
    for run in data_runs:
        prior_array = run["data"]["1"]["prior"]
        collapsed_num_fps = []
        for i in range(len(prior_array)):
            values = []
            for finger_num in run["data"]:
                values.append(run["data"][finger_num]["num_fp"][i])
            collapsed_num_fps.append( sum(values) / len(values) )
        run["data"] = {
            "prior": prior_array,
            "num_fp": collapsed_num_fps
        }
    
    avg_data = data_runs[0].copy()
    errors = []
    for i in range(len(data_runs[0]["data"]["num_fp"])):
        values = []
        for j in range(len(data_runs)):
            run = data_runs[j]
            values.append(run["data"]["num_fp"][i])
        avg = sum(values) / len(values)
        error = sem(values)
        print("For index ", i, " values are ", values, "\naverage = ", avg, "\nsde = ", error)
        avg_data["data"]["num_fp"][i] = avg
        errors.append(error)
    
    #plt.yscale("log")
    plt.xscale("log")
    
    # calculate and plot the baseline
    Xs = avg_data["data"]["prior"].copy()
    baselines = []
    for i in range(len(Xs)):
        baselines.append(1 / Xs[i])
    #plt.plot(Xs, [1.0 for baseline in baselines], marker=markers.pop(), label="exhaustive search")
    # # #
    print(baselines)
    print(avg_data["data"]["num_fp"])
    # calculate and plot the percent reduction in leads required
    Xs = avg_data["data"]["prior"].copy()
    Ys = [ (1 - avg_data["data"]["num_fp"][i] / baselines[i]) * 100 for i in range(len(Xs))]
    errors = [ (errors[i] / baselines[i]) * 100 for i in range(len(errors)) ]
    print("Ys: ", Ys)
    print("Errors: ", errors)
    plt.errorbar(Xs, Ys, errors, marker="o", label="average", capsize=8, color="black", linewidth=2)
    for i in range(len(Xs)):
        plt.text(Xs[i], (Ys[i] + 1.5), round(Ys[i], 1), fontdict={'size'   : 8})

    # # #
    plt.xlabel('Proportion of Fingerprints Belonging to True Criminal')
    plt.ylabel('Percent Reduction in Leads Required')
    plt.title('Relative Forensic Investigation Efficiency')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_root, 'geo_curve_percent.pdf'))
    plt.savefig(os.path.join(output_root, 'geo_curve_percent.png'))
    plt.clf(); plt.close()


# pre: assuming all datafiles are complete and in the proper format
def merge_runs(datafiles):
    output_root = args.output_root
    data_runs = []
    # calculate the average data
    for datafile in datafiles:
        with open(os.path.join(output_root, datafile), "r") as inFile:
            data_runs.append(json.load(inFile))

    avg_data = data_runs[0].copy()
    errors = {finger_num:[] for finger_num in data_runs[0]["data"]}
    for finger_num in data_runs[0]["data"]:
        for i in range(len(data_runs[0]["data"][finger_num]["num_fp"])):
            values = []
            for j in range(len(data_runs)):
                run = data_runs[j]
                values.append(run["data"][finger_num]["num_fp"][i])
            avg = sum(values) / len(values)
            error = sem(values)
            print("For finger num ", finger_num, " and index ", i, " values are ", values, "\naverage = ", avg, "\nsde = ", error)
            avg_data["data"][finger_num]["num_fp"][i] = avg
            errors[finger_num].append(error)
    update_graph(avg_data, errors=errors)

def update_graph(data, errors=None):
    output_root = args.output_root
    markers = ['o', '*', '+', 'x', '^', 's']
    plt.yscale("log")
    plt.xscale("log")
    for finger_num in data["data"]:
        Xs = data["data"][finger_num]["prior"].copy()
        # Convert priors to dataset "size"
        #for i in range(len(Xs)):
        #    Xs[i] = 1.0 / Xs[i]
        Ys = data["data"][finger_num]["num_fp"]
        if errors:
            plt.errorbar(Xs, Ys, errors[finger_num], marker=markers.pop(), label="{} to {}".format(finger_num, finger_num), capsize=5)
        else:
            plt.plot(Xs, Ys, marker=markers.pop(), label="{} to {}".format(finger_num, finger_num))
    # calculate and plot the baseline
    Xs = data["data"]["1"]["prior"].copy()
    baselines = []
    for i in range(len(Xs)):
        baselines.append(1 / Xs[i])
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
        avg /= len(data["data"])
        avgs.insert(0, avg)
    plt.plot(Xs, avgs, label="average", color="black", linewidth=3, zorder=7)
    # # #
    plt.xlabel('Proportion of Fingerprints Belonging to True Criminal')
    plt.ylabel('# of Leads Before Finding True Criminal')
    plt.title('Forensic Investigation Efficiency')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_root, 'geo_curve.pdf'))
    plt.savefig(os.path.join(output_root, 'geo_curve.png'))
    plt.clf(); plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('graph_distribution_shift.py')
    parser.add_argument('--output_root', '-o', nargs='?', help='Root directory for output', \
        const=DEFAULT_OUTPUT_ROOT, default=DEFAULT_OUTPUT_ROOT, type=str)

    args = parser.parse_args()

    print(args)

    percent_reduction_grapher(["geometric_analysis_results_1.json", "geometric_analysis_results_2.json", "geometric_analysis_results_3.json"])

    merge_runs(["geometric_analysis_results_1.json", "geometric_analysis_results_2.json", "geometric_analysis_results_3.json"])

    # TESTED - pass!