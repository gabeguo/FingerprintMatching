import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
import numpy as np

class SiameseFingerprintDataset(Dataset):
    def is_image_filename(self, filename):
        return any(extension in filename for extension in ['.png', \
            '.PNG', '.pneg', '.PNEG', '.jpg', '.JPG', '.jpeg', '.JPEG'])

    """
    Loads the unpaired images from the folder
    1) Stores indexed unpaired images with their corresponding labels in self.images, self.img_labels
    2) Stores all the possible classes in self.classes
    3) Stores images separated by class in self.class_to_images (has image filepath, not actual image)
    """
    def load_images(self):
        self.classes = list() # self.classes[i] = the name of class with index i
        self.img_labels = []
        self.images = []
        self.class_to_images = list()

        for pid in os.listdir(self.root_dir):
            self.classes.append(pid)
            self.class_to_images.append(list())
            curr_person_folder = os.path.join(self.root_dir, pid)
            for sample in os.listdir(curr_person_folder):
                if not self.is_image_filename(sample):
                    continue
                curr_image = os.path.join(curr_person_folder, sample)

                self.img_labels.append(pid)
                self.images.append(curr_image)
                self.class_to_images[-1].append(curr_image)

        return
    """
    Preconditions:
    1) load_images has been called
    Does:
    1) Sanity check desired_num_samples
    2) Sanity check percent_match
    """
    def check_and_set_num_images(self, desired_num_samples, percent_match):
        max_len = len(self.images) * (len(self.images) - 1) // 2
        min_len = 1

        if desired_num_samples is None:
            desired_num_samples = len(self.images) // 2

        desired_num_matching_samples = int(percent_match * desired_num_samples)
        max_possible_matching_samples = sum([len(self.class_to_images[pid]) \
            * (len(self.class_to_images[pid]) - 1) // 2 for pid in range(len(self.class_to_images))])


        if desired_num_samples < min_len or desired_num_samples > max_len:
            raise ValueError('{} desired samples is out of bounds, '.format(desired_num_samples) + \
                'please choose a value between {} and {}'.format(min_len, max_len))
        elif desired_num_matching_samples > max_possible_matching_samples:
            raise ValueError('{}% of matching samples is too high, please choose a lower number'.format(percent_match * 100))
        else: # we good
            self.len = desired_num_samples
            self.percent_match = percent_match
            self.desired_num_matching_samples = desired_num_matching_samples
        return

    """
    Preconditions:
    1) Called self.load_images()
    2) Called self.check_and_set_num_images()
    Does:
    1) Generates pairs of samples, based on self.percent_match and self.desired_num_samples
    2) Samples are in self.pairs, self.pair_labels
    Note: 1 means matching example (from same class), 0 means non-matching example
    """
    def generate_pairs(self):
        self.pairs = []
        self.pair_labels = []
        # get all the matching samples
        # cycle through classes to have most even split
        curr_matching_samples = 0
        counters = [(0,1) for class_index in range(len(self.classes))]
        while curr_matching_samples < self.desired_num_matching_samples:
            # cycle through classes
            for class_index in range(len(self.classes)):
                if curr_matching_samples >= self.desired_num_matching_samples:
                    break

                i, j = counters[class_index]
                curr_class_images = self.class_to_images[class_index]
                # check bounds
                if j >= len(curr_class_images): # second image in pair out of bounds
                    i += 1 # new pair
                    j = i + 1
                if i >= len(curr_class_images): # first image in pair out of bounds, time to move on
                    continue
                # retrieve image
                img1 = curr_class_images[i]
                img2 = curr_class_images[j]
                self.pairs.append([img1, img2])
                self.pair_labels.append(1)
                # increment counters for next time
                counters[class_index] = (i, j + 1)
                # added one more sample
                curr_matching_samples += 1

        # get all the nonmatching samples
        # cycle through the classes to have the most even split
        num_nonmatching_samples = self.len - self.desired_num_matching_samples
        curr_nonmatching_samples = 0
        # a counter for every class pair
        counters = [[(0,0) for j in range(len(self.classes))] \
            for i in range(len(self.classes))]
        while curr_nonmatching_samples < num_nonmatching_samples:
            # cycle through pairs of classes
            for index1 in range(len(self.classes)):
                for index2 in range(index1 + 1, len(self.classes)):
                    # break if we have all the samples we need
                    if curr_nonmatching_samples >= num_nonmatching_samples:
                        break

                    i, j = counters[index1][index2]

                    class1Images = self.class_to_images[index1]
                    class2Images = self.class_to_images[index2]
                    # check bounds
                    if j >= len(class2Images): # second image in pair out of bounds
                        i += 1 # new pair
                        j = 0
                    if i >= len(class1Images): # first image in pair out of bounds, time to move on
                        continue
                    # retrieve image
                    img1 = class1Images[i]
                    img2 = class2Images[j]
                    self.pairs.append([img1, img2])
                    self.pair_labels.append(0)
                    # increment counters for next time
                    counters[index1][index2] = (i, j + 1)
                    # added one more sample
                    curr_matching_samples += 1
        return
    """
    1)
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
    Note: root_dir is not split into train, val, and test
    2)
    percent_match is the desired percent of samples that have the same label
    (as they are returned in pairs, for siamese network training)
    3)
    desired_num_samples is the number of pairs (i.e., len) this dataloader will have.
    It goes anywhere from 1->(num_images choose 2)
    desired_num_samples = None will set len = num_images // 2
    4)
    augmentation tbd

    """
    def __init__(self, root_dir, percent_match=0.5, desired_num_samples=None, augmentation=False):
        # load images
        self.root_dir = root_dir
        self.load_images()

        self.augmentation = augmentation

        # sanity checking for desired_num_samples and percent_match
        self.check_and_set_num_images(desired_num_samples, percent_match)

        self.generate_pairs()

        return

    def __len__(self):
        return self.len

    """
    Returns a pair of images, and whether or not they're in the same class (0, 1)
    """
    def __getitem__(self, idx):
        assert idx < self.len
        actual_images = (read_image(self.pairs[idx][0]), read_image(self.pairs[idx][1]))
        print(self.pairs[idx], self.pair_labels[idx])
        return actual_images, self.pair_labels[idx]
