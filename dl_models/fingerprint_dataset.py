import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np

class FingerprintDataset(Dataset):
    """
    Assumes root_dir has following structure:
    root_dir/
        class1/
            class1sample1.jpg
            class1sample2.jpg
            ...
        class2/
            class2sample1.jpg
            class2sample2.jpg
            ...
        ...
    Note: root_dir is not split into train and test

    Augmentation tbd
    """
    def __init__(self, root_dir, augmentation=False):
        self.root_dir = root_dir
        self.augmentation = augmentation
        self.len = len([x for x in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, x))])
        return

    def __len__(self):
        return self.len
