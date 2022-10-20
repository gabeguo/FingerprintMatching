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

train_features, train_labels = next(iter(train_dataloader))
