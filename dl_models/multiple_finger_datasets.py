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

class MultipleFingerDataset(Dataset):
    """
    Returns triplets of (N_0 anchor fingers, N_1 positive fingers, N_2 negative fingers).
    -> Anchor fingers must be distinct from positive fingers (but can be same as negative).
            Ex: If anchors are right index & right pinky, positive can be left index & left middle
    -> Anchor sensor must be different from positive sensor (but can be same as negative).
    -> All anchor fingers must be from same sensor, all positive fingers must be from same sensor,
       all negative fingers must be from same sensor.
    -> All anchor fingers must be distinct fingers, all positive fingers must be distinct fingers,
       all negative fingers must be distinct fingers
    """

    def __init__(self, fingerprint_dataset, num_anchor_fingers, num_pos_fingers, num_neg_fingers):
        assert num_anchor_fingers + num_pos_fingers <= 10
        assert num_anchor_fingers + num_neg_fingers <= 10
        assert num_anchor_fingers > 0 and num_pos_fingers > 0 and num_neg_fingers > 0

        self.fingerprint_dataset = fingerprint_dataset
        self.train = self.fingerprint_dataset.train
        
        # number of each type of fingers
        self.num_anchor_fingers = num_anchor_fingers
        self.num_pos_fingers = num_pos_fingers
        self.num_neg_fingers = num_neg_fingers

        # initialize the lookup tables
        self.test_labels = self.fingerprint_dataset.test_labels
        self.test_data = self.fingerprint_dataset.test_data
        # generate fixed triplets for testing
        self.labels_set = set(self.test_labels)
        self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0]
                                    for label in self.labels_set}

        self.random_state = np.random.RandomState(29)
        
        # how many times we want to loop through all the items to make triplets
        SCALE_FACTOR = 1

        # Don't allow duplicate combos
        seen_combos = set()

        # Create triplets
        triplets = list()
        for j in range(SCALE_FACTOR):
            for i in range(len(self.test_data)):
                #print('{} out of {}'.format(i, len(self.test_data)))
                while True: # need to find original combos
                    anchor_indices = self.get_anchor_indices(i)
                    positive_indices = self.get_positive_indices(anchor_indices)
                    negative_indices = self.get_negative_indices(anchor_indices)

                    curr_anchor_pos_combo = tuple(sorted(anchor_indices + positive_indices))
                    curr_anchor_neg_combo = tuple(sorted(anchor_indices + negative_indices))
                    if curr_anchor_pos_combo not in seen_combos \
                            and curr_anchor_neg_combo not in seen_combos:
                        seen_combos.add(curr_anchor_pos_combo)
                        seen_combos.add(curr_anchor_neg_combo)
                        break # found original combo in both anchor-positive and anchor-negative

                triplets.append((anchor_indices, positive_indices, negative_indices))
        self.test_triplets = triplets

        return

    """
    Returns a tuple of size self.num_fingers, containing:
    -> base_index
    -> (self.num_fingers - 1) indices that are:
        1) from the same class
        2) from different fingers
        3) from the same sensor
       as base_index
    """
    def get_anchor_indices(self, base_index):
        ret_val = [base_index]

        seen_fgrps = set()
        seen_fgrps.add(self.get_fgrp_from_index(base_index))

        seen_sensors = set()
        seen_sensors.add(self.get_sensor_from_index(base_index))

        while len(ret_val) < self.num_anchor_fingers:
            next_index = base_index
            # guarantees distinct fingers
            while next_index in ret_val or \
                    self.get_fgrp_from_index(next_index) in seen_fgrps or \
                    self.get_sensor_from_index(base_index) != self.get_sensor_from_index(next_index): # guarantees same sensor
                next_index = self.random_state.choice(self.label_to_indices[self.test_labels[base_index]]) # guarantees same class
            
            ret_val.append(next_index)
            seen_fgrps.add(self.get_fgrp_from_index(next_index))
            seen_sensors.add(self.get_sensor_from_index(next_index))
        
        assert list(seen_sensors)[0] == self.get_sensor_from_index(base_index) # ensure 3) same sensor as base_index
        assert len(seen_sensors) == 1 # ensure 3) same sensor (as each other)
        assert len(seen_fgrps) == len(ret_val) # ensure 2) different fingers
        assert len(set([self.test_labels[i] for i in ret_val])) == 1 # ensure 1) same class
        
        return tuple(ret_val)

    """
    Returns a tuple of size self.num_pos_fingers that are:
    1) from same class as anchor_indices
    2) from different fingers than anchor_indices
    3) from different sensor than anchor_indices
    4) from different fingers than each other
    5) from same sensor as each other
    """
    def get_positive_indices(self, anchor_indices):
        ret_val = []
        seen_fgrps = set([self.get_fgrp_from_index(i) for i in anchor_indices])

        # get first positive example
        first_pos_index = anchor_indices[0] # 1) ensure same class as anchor_indices
        # 2) ensure different finger than anchor, 3) different sensor than anchor
        while first_pos_index in anchor_indices \
                or self.get_fgrp_from_index(first_pos_index) in seen_fgrps \
                or self.get_sensor_from_index(first_pos_index) == self.get_sensor_from_index(anchor_indices[0]):
            # 1) ensure same class as anchor_indices
            first_pos_index = self.random_state.choice(self.label_to_indices[self.test_labels[first_pos_index]])
        ret_val.append(first_pos_index)
        seen_fgrps.add(self.get_fgrp_from_index(first_pos_index))
        the_sensor = self.get_sensor_from_index(first_pos_index)

        while len(ret_val) < self.num_pos_fingers:
            pos_index = first_pos_index
            # 2) ensure different fingers than anchor_indices, 4) than each other, 5) same sensor as each other
            while pos_index in ret_val \
                    or pos_index in anchor_indices \
                    or self.get_fgrp_from_index(pos_index) in seen_fgrps \
                    or self.get_sensor_from_index(pos_index) != the_sensor:
                # 1) ensure same class as anchor_indices
                pos_index = self.random_state.choice(self.label_to_indices[self.test_labels[first_pos_index]])
            ret_val.append(pos_index)
            seen_fgrps.add(self.get_fgrp_from_index(pos_index)) # track used fingers
        
        assert len(set([self.get_sensor_from_index(i) for i in ret_val])) == 1 # ensure 5) same sensor as each other
        assert self.get_sensor_from_index(ret_val[0]) != self.get_sensor_from_index(anchor_indices[0]) # ensure 3) different sensor than anchor
        assert len(seen_fgrps) == len(ret_val) + len(anchor_indices) # ensure 2) different fingers than anchor indices, 4) each other
        assert self.test_labels[ret_val[-1]] == self.test_labels[anchor_indices[-1]] # ensure 1) same class as anchor_indices
        assert len(set([self.test_labels[i] for i in ret_val])) == 1 # ensure 1) same class as anchor_indices (and each other)
        
        return tuple(ret_val)

    """
    Returns a tuple of size self.num_neg_fingers that are:
    1) from different class than anchor_indices
    2) from different fingers than each other
    3) from same sensor as each other
    4) from same class as each other
    """
    def get_negative_indices(self, anchor_indices):
        ret_val = []
        seen_fgrps = set()
        seen_sensors = set()

        # ensures 1) different class than anchor_indices
        neg_label = np.random.choice(
            list(self.labels_set - set([self.test_labels[anchor_indices[0]]]))
        )
        
        first_neg_index = self.random_state.choice(self.label_to_indices[neg_label])
        ret_val.append(first_neg_index)
        seen_fgrps.add(self.get_fgrp_from_index(first_neg_index))
        seen_sensors.add(self.get_sensor_from_index(first_neg_index))

        while len(ret_val) < self.num_neg_fingers:
            neg_index = first_neg_index
            # ensures 2) different fingers than each other, 3) same sensor as each other
            while neg_index in ret_val \
                    or self.get_fgrp_from_index(neg_index) in seen_fgrps \
                    or self.get_sensor_from_index(neg_index) not in seen_sensors:
                # ensures 4) same class as each other
                neg_index = self.random_state.choice(self.label_to_indices[neg_label])
            
            ret_val.append(neg_index)
            seen_fgrps.add(self.get_fgrp_from_index(neg_index))
            seen_sensors.add(self.get_sensor_from_index(neg_index))
        
        assert len(set([self.test_labels[i] for i in ret_val])) == 1 # ensure 4) same class as each other
        assert self.test_labels[ret_val[0]] != self.test_labels[anchor_indices[0]] # ensure 1) different class as anchor
        assert len(seen_sensors) == 1 # ensure 3) same sensor as each other
        assert len(seen_fgrps) == len(ret_val)  # ensure 2) different fgrps from each other
        
        return tuple(ret_val)
    """
    only works on the server: assumes that paths have format like:
    /data/therealgabeguo/fingerprint_data/sd300a_split/train/00001765/00001765_plain_500_08.png
    """
    def get_dataset_name(self, filepath):
        DATASET_NAME_INDEX = 4
        ret_val = filepath.split('/')[DATASET_NAME_INDEX]
        assert 'sd30' in ret_val or 'RidgeBase' in ret_val or 'SOCOFing' in ret_val
        return ret_val

    def get_filename_from_index(self, i):
        return self.test_data[i].split('/')[-1]
    
    def get_sensor_from_index(self, i):
        return get_sensor(self.get_filename_from_index(i))

    def get_fgrp_from_index(self, i):
        return get_fgrp(self.get_filename_from_index(i))

    """
    returns: 
    1) triplet of tuples of images, where:
        a) first tuple is anchor, second tuple is positive, third tuple is negative
        b) first tuple has size self.num_anchor_fingers, 
            second tuple has size self.num_pos_fingers, third tuple has size self.num_neg_fingers
    2) triplet of class labels corresponding to images (not tuple, since in each image tuple, all labels are same)
    3) triplet of tuples of filepaths corresponding to images
    """
    def __getitem__(self, index):
        anchor_filepaths = [self.test_data[index] for index in self.test_triplets[index][0]]
        pos_filepaths = [self.test_data[index] for index in self.test_triplets[index][1]]
        neg_filepaths = [self.test_data[index] for index in self.test_triplets[index][2]]

        the_labels = [self.test_labels[self.test_triplets[index][0][0]], \
                      self.test_labels[self.test_triplets[index][1][0]], \
                      self.test_labels[self.test_triplets[index][2][0]]]
        assert the_labels[0] == the_labels[1]
        assert the_labels[0] != the_labels[2]

        # TODO: support .bmp
        anchor_imgs = [my_transformation(read_image(x, mode=ImageReadMode.RGB), train=self.train) for x in anchor_filepaths]
        pos_imgs = [my_transformation(read_image(x, mode=ImageReadMode.RGB), train=self.train) for x in pos_filepaths]
        neg_imgs = [my_transformation(read_image(x, mode=ImageReadMode.RGB), train=self.train) for x in neg_filepaths]
        
        return (anchor_imgs, pos_imgs, neg_imgs), the_labels, (anchor_filepaths, pos_filepaths, neg_filepaths)

    def __len__(self):
        if not self.train: # we can have multiple testing triplets for each item in the dataset
            return len(self.test_triplets)
        return len(self.fingerprint_dataset)


