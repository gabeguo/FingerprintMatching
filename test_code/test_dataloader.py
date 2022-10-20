import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import os

sys.path.append('../dl_models')
sys.path.append('../')

from fingerprint_dataset import SiameseFingerprintDataset
from common_filepaths import DATA_FOLDER

training_data = SiameseFingerprintDataset(os.path.join(DATA_FOLDER, 'train'))
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

num_positive_examples = 0
for i in range(30):#range(len(training_data)):
    train_images, train_labels, train_filenames = next(iter(train_dataloader))
    num_positive_examples += train_labels.item()
    print(train_filenames, train_labels)

print(num_positive_examples)
