import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision.io import read_image, ImageReadMode
from torchvision import models, transforms

import torchvision.transforms.functional as F

import sys
sys.path.append('../directory_organization')
from fileProcessingUtil import get_id, get_fgrp, get_sensor

# Use https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10
# makes images squares by padding
class SquarePad:
    def __init__(self, fill_val):
        assert fill_val <= 255 and fill_val >= 0
        self.fill_val = fill_val
        return
    def __call__(self, image):
        max_wh = max(image.size())
        p_left, p_top = [(max_wh - s) // 2 for s in image.size()[1:]] # first channel is just colors
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size()[1:], [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, self.fill_val, 'constant')

# returns the image as a normalized square with standard size
def my_transformation(the_image, train=False, target_image_size=(224, 224)):
    #print(target_image_size)
    assert target_image_size[0] == target_image_size[1]
    fill_val = 255 if the_image[0, 0, 0] > 200 else 0
    # common transforms - these are the only transforms for test
    transform=transforms.Compose([
        SquarePad(fill_val=fill_val),
        transforms.Resize(target_image_size),
        transforms.Grayscale(num_output_channels=3),
        # transforms.RandomInvert(p=1.0), # TODO: remove this after the experiment
        transforms.Normalize([0, 0, 0], [1, 1, 1]),
        #transforms.Normalize([210, 210, 210], [70, 70, 70]),
    ])
    if train and torch.rand(1).item() < 0.65: # randomly apply the train transforms
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=12.5, translate=(0.075, 0.075), scale=(0.925, 1.075), shear=(-7.5, 7.5), fill=fill_val),
            transform,
            transforms.RandomResizedCrop(size=target_image_size, scale=(0.9, 1), ratio=(0.95, 1.05)),
            #transforms.RandomInvert(p=0.3),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.01, 0.75)),
        ])
        the_image = transform(the_image.float())
        # add noise
        shadeScaling = 1 + 0.3 * (torch.rand(1).item() - 0.5) # shade scaling between 0.85 and 1.15
        noise = 0.1 * torch.max(the_image) * (torch.rand(target_image_size) - 0.5)
        the_image = shadeScaling * (the_image + noise)
        return the_image
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
            #print(len(self.labels_set), 'labels')

        else:
            # initialize the lookup tables
            self.test_labels = self.fingerprint_dataset.test_labels
            self.test_data = self.fingerprint_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)
            
            # constants
            POS = 0
            NEG = 1

            # how many times we want to loop through all the items to make triplets
            SCALE_FACTOR = 2

            # implement balanced number of each fingerprint type pair, e.g., index, pinky
            count_per_pair = SCALE_FACTOR * int(len(self.test_labels) // (10 * 10 // 2) * 1.1) # give a bit of slack (not exactly even)
            desired_num_finger_pairs = np.full((11, 11, 2), count_per_pair)
            curr_num_finger_pairs = np.zeros((11, 11, 2))

            #print(count_per_pair)

            # Don't allow duplicate pairs
            seen_pairs = set()

            # Create balanced triplets
            triplets = list()
            
            for j in range(SCALE_FACTOR):
                for i in range(len(self.test_data)):
                    #print('{} out of {}'.format(i, len(self.test_data)))
                    while True: # do until we find a triplet that doesn't exceed capacity
                        anchor_index = i
                        pos_index = random_state.choice(self.label_to_indices[self.test_labels[i]])
                        neg_index = random_state.choice(self.label_to_indices[
                            np.random.choice(
                                list(self.labels_set - set([self.test_labels[i]]))
                            )
                        ])
                        curr_triplet = [anchor_index, pos_index, neg_index]
                        filepaths = (self.test_data[anchor_index], self.test_data[pos_index], self.test_data[neg_index])
                        anchor_fname, pos_fname, neg_fname = (the_filepath.split('/')[-1] for the_filepath in filepaths)
                        anchor_fgrp, pos_fgrp, neg_fgrp = int(get_fgrp(anchor_fname)), int(get_fgrp(pos_fname)), int(get_fgrp(neg_fname))

                        # we can still add more if:
                        # 1) it hasn't reached desired number yet
                        # 2) this sample combo hasn't been seen yet
                        if curr_num_finger_pairs[anchor_fgrp, pos_fgrp, POS] < desired_num_finger_pairs[anchor_fgrp, pos_fgrp, POS] \
                                and curr_num_finger_pairs[anchor_fgrp, neg_fgrp, NEG] < desired_num_finger_pairs[anchor_fgrp, neg_fgrp, NEG] \
                                and (anchor_index, pos_index) not in seen_pairs and (anchor_index, neg_index) not in seen_pairs:
                            # update counts
                            curr_num_finger_pairs[anchor_fgrp, pos_fgrp, POS] += 1
                            curr_num_finger_pairs[pos_fgrp, anchor_fgrp, POS] += 1 # get transpose too
                            curr_num_finger_pairs[anchor_fgrp, neg_fgrp, NEG] += 1
                            curr_num_finger_pairs[neg_fgrp, anchor_fgrp, NEG] += 1
                            
                            # we can use this triplet
                            triplets.append(curr_triplet) 
                            # mark these samples as used
                            seen_pairs.update([(anchor_index, pos_index), (pos_index, anchor_index),\
                                                (anchor_index, neg_index), (neg_index, anchor_index)]) 
                            break # move on to next index
            #print(curr_num_finger_pairs[:,:,POS])
            #print(curr_num_finger_pairs[:,:,NEG])

            # Deprecated: unbalanced triplets
            # triplets = [[i,
            #              random_state.choice(self.label_to_indices[self.test_labels[i]]),
            #              random_state.choice(self.label_to_indices[
            #                                      np.random.choice(
            #                                          list(self.labels_set - set([self.test_labels[i]]))
            #                                      )
            #                                  ])
            #              ]
            #             for i in range(len(self.test_data))]
            self.test_triplets = triplets

    """
    only works on the server: assumes that paths have format like:
    /data/therealgabeguo/fingerprint_data/sd300a_split/train/00001765/00001765_plain_500_08.png
    """
    def get_dataset_name(self, filepath):
        DATASET_NAME_INDEX = 4
        ret_val = filepath.split('/')[DATASET_NAME_INDEX]
        assert 'sd30' in ret_val
        return ret_val

    # returns image triplet, class labels of images, filepaths of images
    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index]
            #print(img1)
            #print('anchor example', label1, get_sensor(img1.split('/')[-1]))
            #print(label1)

            anchor_dataset = self.get_dataset_name(img1)
            #print('anchor example', index, anchor_dataset)

            # choose positive example, not from same sensor
            positive_index = index
            while positive_index == index \
                    or anchor_dataset != self.get_dataset_name(self.train_data[positive_index]) \
                    or get_sensor(img1.split('/')[-1]) == get_sensor(self.train_data[positive_index].split('/')[-1]): 
                    # make sure positive example doesn't come from same sensor, so it doesn't learn background
                    # positive example also has to be from same dataset
                positive_index = np.random.choice(self.label_to_indices[label1])
            #print('\tpositive example', index, self.get_dataset_name(self.train_data[positive_index]))
            #print('\tpositive example', self.train_labels[positive_index], get_sensor(self.train_data[positive_index].split('/')[-1]))

            # choose negative example
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            while anchor_dataset != self.get_dataset_name(self.train_data[self.label_to_indices[negative_label][0]]):
                # the negative class isn't even from same dataset
                negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            while anchor_dataset != self.get_dataset_name(self.train_data[negative_index]):
                print('shouldnt be running')
                raise ValueError()
                # negative example has to come from same dataset, so it's not too easy
                negative_index = np.random.choice(self.label_to_indices[negative_label])
            #print('\tnegative example', index, self.get_dataset_name(self.train_data[negative_index]))
            #print('\tnegative example', negative_label, get_sensor(self.train_data[negative_index].split('/')[-1]))
            """
            while get_sensor(img1.split('/')[-1]) == get_sensor(self.train_data[negative_index].split('/')[-1]):
                #print(negative_index, get_sensor(img1.split('/')[-1]), get_sensor(self.train_data[negative_index].split('/')[-1]))
                negative_index = np.random.choice(self.label_to_indices[negative_label])
            #print('\tnegative example', negative_label, get_sensor(self.train_data[negative_index].split('/')[-1]))
            """

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
        if not self.train: # we can have multiple testing triplets for each item in the dataset
            return len(self.test_triplets)
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
