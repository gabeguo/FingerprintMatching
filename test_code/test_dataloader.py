import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(1, '../dl_models')
sys.path.insert(1, '../')

from fingerprint_dataset import SiameseFingerprintDataset
import common_filepaths

training_data = SiameseFingerprintDataset(os.path.join(DATA_FOLDER, 'train'))
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
