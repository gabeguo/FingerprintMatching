import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils
from torchvision.io import read_image, ImageReadMode
import numpy as np
import torchvision.transforms.functional as F
from siamese_datasets import my_transformation

class FingerprintDataset(Dataset):
    def is_image_filename(self, filename):
        return any(extension in filename for extension in ['.png', \
            '.PNG', '.pneg', '.PNEG', '.jpg', '.JPG', '.jpeg', '.JPEG'])

    """
    Loads the unpaired images from the folder(s)
    1) Stores indexed unpaired images with their corresponding labels in self.images, self.img_labels
    2) Stores all the possible classes in self.classes
    3) Stores images separated by class in self.class_to_images (has image filepath, not actual image)

    Note: root_dirs can be a string with a singular root directory, or a list of strings
    """
    def load_images(self, root_dirs):
        self.classes = list() # self.classes[i] = the name of class with index i
        self.img_labels = list()
        self.images = list()
        self.class_to_images = list()

        if type(root_dirs) is not list:
            root_dirs = [root_dirs]
        # can have multiple data sources
        for root_dir in root_dirs:
            # go through all people
            for pid in os.listdir(root_dir):
                self.classes.append(pid)
                self.class_to_images.append(list())
                curr_person_folder = os.path.join(root_dir, pid)
                # go through all images
                for sample in os.listdir(curr_person_folder):
                    if not self.is_image_filename(sample):
                        continue
                    curr_image = os.path.join(curr_person_folder, sample)
                    # in testing mode, we can't have duplicate images (i.e., 500, 1000, 2000 res versions of image)
                    if (not self.train) and any(resolution in curr_image for resolution in ['_1000_', '_2000_']): 
                        continue
                    # all good, add the image
                    self.img_labels.append(pid)
                    self.images.append(curr_image)
                    self.class_to_images[-1].append(curr_image)

        self.len = len(self.img_labels)

        return

    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train

        self.load_images(root_dir)

        if self.train:
            self.train_labels = self.img_labels
            self.train_data = self.images
        else: # test
            self.test_labels = self.img_labels
            self.test_data = self.images

        return

    def __len__(self):
        return self.len

    # THIS IS NOT CALLED FROM TRIPLET DATASET
    # returns image, label, filepath
    def __getitem__(self, idx):
        # TODO: Add data augmentation
        return my_transformation(read_image(self.images[idx], mode=ImageReadMode.RGB), train=self.train), \
        self.img_labels[idx], \
        self.images[idx]