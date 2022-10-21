import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import os

sys.path.append('../dl_models')
sys.path.append('../siamese-triplet')
sys.path.append('../')

from fingerprint_dataset import FingerprintDataset
from datasets import SiameseDataset, TripletDataset
from common_filepaths import DATA_FOLDER

print('pair loading test\n')

training_dataset = SiameseDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'train'), train=True))
train_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)

num_positive_examples = 0
for i in range(30):#range(len(training_data)):
    train_images, train_label, train_filepaths = next(iter(train_dataloader))
    num_positive_examples += train_label
    print(train_filepaths, train_label)

print(num_positive_examples)

print('\ntriplet loading test\n')

triplet_dataset = TripletDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'train'), train=True))
triplet_dataloader = DataLoader(triplet_dataset, batch_size=1, shuffle=True)

num_positive_examples = 0
for i in range(30):#range(len(training_data)):
    test_images, test_labels, test_filepaths = next(iter(triplet_dataloader))
    print(test_filepaths, test_labels)


# TEST PASSED :)
