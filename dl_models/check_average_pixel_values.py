import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import sys
import os
import math

sys.path.append('../')
sys.path.append('../directory_organization')

from trainer import *
from losses import *
from siamese_datasets import *
from fingerprint_dataset import *
from embedding_models import *
from fileProcessingUtil import *

from common_filepaths import DATA_FOLDER, SUBSET_DATA_FOLDER, BALANCED_DATA_FOLDER, UNSEEN_DATA_FOLDER, EXTRA_DATA_FOLDER, \
    ENHANCED_DATA_FOLDER, ENHANCED_HOLDOUT_FOLDER, ENHANCED_INK_FOLDER, SYNTHETIC_DATA_FOLDER

# this should do nothing
supposedly_useless_normalization = transforms.Normalize([0, 0, 0], [1, 1, 1])

# Data loading 
batch_size=8
for the_data_folder in [DATA_FOLDER, UNSEEN_DATA_FOLDER, EXTRA_DATA_FOLDER, SYNTHETIC_DATA_FOLDER]:
    the_dataset = FingerprintDataset(os.path.join(the_data_folder, 'test'), train=False)
    print('loaded test dataset: {}'.format(the_data_folder))
    test_dataloader = DataLoader(the_dataset, batch_size=batch_size, shuffle=True)

    # import matplotlib.pyplot as plt
    mins, maxs, avgs, stds = [], [], [], []
    norm_mins, norm_maxs, norm_avgs, norm_stds = [], [], [], []
    it = iter(test_dataloader)
    for i in range(20):
        processed_images, labels, filepaths = next(it)
        if i == 0:
            print('\timage batch shape:', processed_images.size())
            print('\tlabels shape:', len(labels))
            print('\tfilepaths shape:', len(filepaths))
        for item in range(len(labels)):
            raw_image = read_image(filepaths[0], mode=ImageReadMode.RGB).float()
            fake_normalized_image = supposedly_useless_normalization(raw_image) # this should be the same

            the_min = torch.min(raw_image)
            the_max = torch.max(raw_image)
            the_avg = torch.mean(raw_image)
            the_std = torch.std(raw_image)
            # print('min: {}\nmax: {}\navg: {}\nstd: {}'.format(\
            #     the_min, the_max, the_avg, the_std))
            mins.append(the_min)
            maxs.append(the_max)
            avgs.append(the_avg)
            stds.append(the_std)

            norm_mins.append(torch.min(fake_normalized_image))
            norm_maxs.append(torch.max(fake_normalized_image))
            norm_avgs.append(torch.mean(fake_normalized_image))
            norm_stds.append(torch.std(fake_normalized_image))
        # next_img = (next_img - the_min) / (the_max - the_min)
        # print(next_img[0])
        # plt.imshow(next_img.permute(1, 2, 0))
        # plt.show()
    print('\taverage min:', np.mean(mins))
    print('\t\taverage min:', np.mean(norm_mins))
    print('\taverage max:', np.mean(maxs))
    print('\t\taverage max:', np.mean(norm_maxs))
    print('\taverage avg:', np.mean(avgs))
    print('\t\taverage avg:', np.mean(norm_avgs))
    print('\taverage std:', np.mean(stds))
    print('\t\taverage std:', np.mean(norm_stds))