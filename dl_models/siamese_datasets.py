import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision.io import read_image, ImageReadMode
from torchvision import models, transforms

import torchvision.transforms.functional as F

# Use https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10
# makes images squares by padding
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size())
        p_left, p_top = [(max_wh - s) // 2 for s in image.size()[1:]] # first channel is just colors
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size()[1:], [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 255, 'constant')

# returns the image as a normalized square with standard size
def my_transformation(the_image, train=False, target_image_size=(224, 224)):
    #print(target_image_size)
    assert target_image_size[0] == target_image_size[1]
    transform=transforms.Compose([
        SquarePad(),
        transforms.Resize(target_image_size),
        #transforms.Normalize([0, 0, 0], [1, 1, 1]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if train:
        transform = transforms.Compose([
            transforms.RandomRotation(30, fill=255),
            transforms.RandomAffine(degrees=5, shear=(-5, 5, -5, 5), fill=255),
            #transforms.ColorJitter(),
            transform,
            #transforms.Resize((int(target_image_size[0] * 1.1), int(target_image_size[1] * 1.1))),
            #transforms.RandomResizedCrop(target_image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            #transforms.RandomHorizontalFlip(p=0.25),
            #transforms.RandomVerticalFlip(p=0.25),
            #transforms.RandomAdjustSharpness(1.5, p=0.25),
        ])
    #the_min = torch.min(the_image)
    #the_max = torch.max(the_image)
    #the_image = (the_image - the_min) / (the_max - the_min)
    return transform(the_image.float())

class SiameseDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
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
            # generate fixed pairs for testing
            self.test_labels = self.fingerprint_dataset.test_labels
            self.test_data = self.fingerprint_dataset.test_data
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i]]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    # returns image pair, label (pos or neg), filepaths of images
    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        filepath1, filepath2 = img1, img2
        img1 = my_transformation(read_image(img1, mode=ImageReadMode.RGB), train=self.train)
        img2 = my_transformation(read_image(img2, mode=ImageReadMode.RGB), train=self.train)

        return (img1, img2), target, (filepath1, filepath2)

    def __len__(self):
        return len(self.fingerprint_dataset)


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


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
