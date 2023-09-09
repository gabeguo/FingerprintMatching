import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

import sys
sys.path.append('../dl_models')
from dl_models.multiple_finger_datasets import my_transformation

class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, fingerprint_dataset):
        self.fingerprint_dataset = fingerprint_dataset
        self.train = self.fingerprint_dataset.train

        if self.train:
            self.train_labels = self.fingerprint_dataset.train_labels
            self.train_data = self.fingerprint_dataset.train_data
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.array(self.train_labels) == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.fingerprint_dataset.test_labels
            self.test_data = self.fingerprint_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    # returns image triplet, class labels of images, filepaths of images
    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]

            the_labels = [label1, label1, negative_label]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

            the_labels = []

        filepath1, filepath2, filepath3 = img1, img2, img3

        img1 = my_transformation(read_image(img1, mode=ImageReadMode.RGB), train=self.train)
        img2 = my_transformation(read_image(img2, mode=ImageReadMode.RGB), train=self.train)
        img3 = my_transformation(read_image(img3, mode=ImageReadMode.RGB), train=self.train)

        return (img1, img2, img3), [], (filepath1, filepath2, filepath3)

    def __len__(self):
        return len(self.fingerprint_dataset)

