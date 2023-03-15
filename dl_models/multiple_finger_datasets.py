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

ALL_FINGERS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

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

def my_read_image(x):
    if '.bmp' in x:
        pil2tensor = transforms.Compose([transforms.PILToTensor()]) # for .bmp images
        return pil2tensor(Image.open(x).convert('RGB'))
    return read_image(x, mode=ImageReadMode.RGB)

class MultipleFingerDataset(Dataset):
    """
    Returns triplets of (N_0 anchor fingers, N_1 positive fingers, N_2 negative fingers).
    -> Anchor fingers must be distinct from positive and negative fingers (unless doing 1-1 correlation test)
            Ex: If anchors are right index & right pinky, positive can be left index & left middle, 
            negative can be left index & left middle.
    -> Anchor sensor must be different from positive sensor.
    -> All anchor fingers must be from same sensor, all positive fingers must be from same sensor,
       all negative fingers must be from same sensor.
    -> All anchor fingers must be distinct fingers, all positive fingers must be distinct fingers,
       all negative fingers must be distinct fingers
    -> We can only have fingers from acceptable_anchor_fgrps, acceptable_pos_fgrps, acceptable_neg_fgrps
    """

    def __init__(self, fingerprint_dataset, num_anchor_fingers, num_pos_fingers, num_neg_fingers, \
                SCALE_FACTOR=1, \
                diff_fingers_across_sets=True, diff_fingers_within_set=True, \
                diff_sensors_across_sets=True, same_sensor_within_set=True, \
                acceptable_anchor_fgrps=ALL_FINGERS, acceptable_pos_fgrps=ALL_FINGERS, acceptable_neg_fgrps=ALL_FINGERS):
        if diff_fingers_across_sets and diff_fingers_within_set:
            assert num_anchor_fingers + num_pos_fingers <= 10
            assert num_anchor_fingers + num_neg_fingers <= 10
        assert num_anchor_fingers > 0 and num_pos_fingers > 0 and num_neg_fingers > 0

        self.fingerprint_dataset = fingerprint_dataset
        self.train = self.fingerprint_dataset.train
        if self.train:
            assert num_anchor_fingers == 1 and num_pos_fingers == 1 and num_neg_fingers == 1
        
        # number of each type of fingers
        self.num_anchor_fingers = num_anchor_fingers
        self.num_pos_fingers = num_pos_fingers
        self.num_neg_fingers = num_neg_fingers

        # finger acceptability rules
        self.diff_fingers_across_sets = diff_fingers_across_sets
        self.diff_fingers_within_set = diff_fingers_within_set
        self.diff_sensors_across_sets = diff_sensors_across_sets
        self.same_sensor_within_set = same_sensor_within_set

        self.acceptable_anchor_fgrps = acceptable_anchor_fgrps
        self.acceptable_pos_fgrps = acceptable_pos_fgrps
        self.acceptable_neg_fgrps = acceptable_neg_fgrps

        # number of times to go through dataset in testing
        self.scale_factor = SCALE_FACTOR

        # initialize the lookup tables
        self.the_labels = self.fingerprint_dataset.train_labels if self.train else self.fingerprint_dataset.test_labels
        self.the_data = self.fingerprint_dataset.train_data if self.train else self.fingerprint_dataset.test_data
        # generate fixed triplets for testing
        self.labels_set = set(self.the_labels)
        self.label_to_indices = {label: np.where(np.array(self.the_labels) == label)[0]
                                    for label in self.labels_set}

        self.random_state = np.random.RandomState(29)

        if self.train:
            self.choose_train_anchors()
        else:
            self.choose_test_triplets()

        return

    """
    Chooses the fingers to use as anchors for training, sets self.train_anchor_indices
    """
    def choose_train_anchors(self):
        self.train_anchor_indices = list()
        # ignore scale factor, since pos and neg are randomized every time
        for i in range(len(self.the_data)):
            if self.get_fgrp_from_index(i) in self.acceptable_anchor_fgrps:
                self.train_anchor_indices.append(i)
        return

    """
    Sets self.test_triplets, according to rules laid out in __init__
    """
    def choose_test_triplets(self):
        # Don't allow duplicate combos
        seen_combos = set()

        # Create triplets
        triplets = list()
        for j in range(self.scale_factor):
            for i in range(len(self.the_data)):
                #print('{} out of {}'.format(i, len(self.the_data)))
                while True: # need to find original combos
                    if self.get_fgrp_from_index(i) not in self.acceptable_anchor_fgrps:
                        break # this will never give a valid triplet
                    anchor_indices = self.get_anchor_indices(i, \
                        diff_fingers_within_set=self.diff_fingers_within_set, same_sensor_within_set=self.same_sensor_within_set,
                        possible_fgrps=self.acceptable_anchor_fgrps)
                    positive_indices = self.get_indices(anchor_indices, same_class_as_anchor=True, \
                        diff_fingers_across_sets=self.diff_fingers_across_sets, diff_fingers_within_set=self.diff_fingers_within_set, \
                        diff_sensors_across_sets=self.diff_sensors_across_sets, same_sensor_within_set=self.same_sensor_within_set,
                        possible_fgrps=self.acceptable_pos_fgrps)
                    negative_indices = self.get_indices(anchor_indices, same_class_as_anchor=False, \
                        diff_fingers_across_sets=self.diff_fingers_across_sets, diff_fingers_within_set=self.diff_fingers_within_set, \
                        diff_sensors_across_sets=self.diff_sensors_across_sets, same_sensor_within_set=self.same_sensor_within_set,
                        possible_fgrps=self.acceptable_neg_fgrps)

                    curr_anchor_pos_combo = tuple(sorted(anchor_indices + positive_indices))
                    curr_anchor_neg_combo = tuple(sorted(anchor_indices + negative_indices))
                    if curr_anchor_pos_combo not in seen_combos \
                            and curr_anchor_neg_combo not in seen_combos:
                        seen_combos.add(curr_anchor_pos_combo)
                        seen_combos.add(curr_anchor_neg_combo)
                        # We found someting! We can add it
                        triplets.append((anchor_indices, positive_indices, negative_indices))
                        break # found original combo in both anchor-positive and anchor-negative
        self.test_triplets = triplets
        return

    """
    Returns a tuple of size self.num_fingers, containing:
    -> base_index
    -> (self.num_fingers - 1) indices that are:
        1) different samples from each other
        2) from the same class
        3) (optionally) from different fingers
        4) (optionally) from the same sensor
        5) from certain fingers, as told by possible_fgrps
        6) from same dataset
       as base_index
    """
    def get_anchor_indices(self, base_index, diff_fingers_within_set=True, same_sensor_within_set=True,\
                           possible_fgrps=ALL_FINGERS):
        # TODO: debug possible_fgrps
        ret_val = [base_index]

        seen_fgrps = set()
        seen_fgrps.add(self.get_fgrp_from_index(base_index))

        seen_sensors = set()
        seen_sensors.add(self.get_sensor_from_index(base_index))

        while len(ret_val) < self.num_anchor_fingers:
            # guarantees (2) same class
            next_index = self.random_state.choice(self.label_to_indices[self.the_labels[base_index]]) # guarantees same class
            # guarantees (1) distinct samples
            if next_index in ret_val:
                continue
            # guarantees (3) different fingers (optional)
            if diff_fingers_within_set and (self.get_fgrp_from_index(next_index) in seen_fgrps):
                continue
            # guarantees (4) same sensor (optional)
            if same_sensor_within_set and (self.get_sensor_from_index(base_index) != self.get_sensor_from_index(next_index)):
                continue
            # guarantees (5) only from certain fingers
            if self.get_fgrp_from_index(next_index) not in possible_fgrps:
                continue
            # guarantees (6) from same dataset
            if self.get_datasetName_from_index(next_index) != self.get_datasetName_from_index(base_index):
                continue
                
            ret_val.append(next_index) # ensure (1) different samples
            seen_fgrps.add(self.get_fgrp_from_index(next_index)) # ensure (3) different fingers (optional)
            seen_sensors.add(self.get_sensor_from_index(next_index)) # ensure (4) same sensor (optional)
        
        if same_sensor_within_set:
            assert list(seen_sensors)[0] == self.get_sensor_from_index(base_index) # ensure 4) same sensor as base_index
        if same_sensor_within_set:
            assert len(seen_sensors) == 1 # ensure 4) same sensor (as each other) (optional)
        if diff_fingers_within_set:
            assert len(seen_fgrps) == len(ret_val) # ensure 3) different fingers (optional)
        assert len(set([self.the_labels[i] for i in ret_val])) == 1 # ensure 2) same class
        assert len(set(ret_val)) == len(ret_val) # ensure 1) distinct samples
        assert seen_fgrps.issubset(set(possible_fgrps)) # ensure 5) only from certain fingers
        assert len(set([self.get_datasetName_from_index(i) for i in ret_val])) == 1 # ensure 6) only one dataset
        
        return tuple(ret_val)

    """
    Returns a tuple that:
    (0) definitely same class as each other
    (1) from same (a) / diff (b) class as anchor_indices, depending on same_class_as_anchor
    (2) size self.num_pos_fingers (a) or self.num_neg_fingers (b), depending on same_class_as_anchor
    (3) definitely distinct samples from anchor_indices
    (4) definitely distinct samples from each other
    (5) (optionally) from different fingers than anchor_indices
    (6) (optionally) from different fingers than each other
    (7) (optionally) from different sensor than anchor_indices
    (8) (optionally) from same sensor as each other
    (9) has only fingers from possible_fgrps
    (10) from same dataset as anchor_indices
    """
    def get_indices(self, anchor_indices, same_class_as_anchor, \
                    diff_fingers_across_sets=True, diff_fingers_within_set=True, \
                    diff_sensors_across_sets=True, same_sensor_within_set=True, \
                    possible_fgrps=ALL_FINGERS):
        # TODO: debug possible_fgrps
        ret_val = []

        # satisfy (5) - different fingers than anchor (possibly)
        anchor_fgrps = set([self.get_fgrp_from_index(i) for i in anchor_indices])
        retVal_fgrps = set()
        # satisfy (7) - different sensors than anchor (possibly)
        anchor_sensors = set([self.get_sensor_from_index(i) for i in anchor_indices])
        retVal_sensors = set()

        # satisfy (1)(a), (2)(a) - same class as anchor
        if same_class_as_anchor:
            the_label = self.the_labels[anchor_indices[0]]
            the_size = self.num_pos_fingers
        # satisfy (1)(b), (2)(b) - diff class than anchor
        else:
            while True:
                the_label = np.random.choice(
                    list(self.labels_set - set([self.the_labels[anchor_indices[0]]]))
                )
                random_index = self.random_state.choice(self.label_to_indices[the_label])
                if self.get_datasetName_from_index(random_index) == self.get_datasetName_from_index(anchor_indices[0]):
                    break # satisfy (10) - same dataset as anchors
            the_size = self.num_neg_fingers
        
        while len(ret_val) < the_size:
            # satisfy (0) - same class as each other
            curr_index = self.random_state.choice(self.label_to_indices[the_label])
            """
            print('anchors: {}'.format([self.get_filename_from_index(the_idx) for the_idx in anchor_indices]))
            print('selected: {}'.format(self.get_filename_from_index(curr_index)))
            """
            if curr_index in anchor_indices or curr_index in ret_val:
                continue # satisfy (3), (4) - try again until we get previously unseen samples
            curr_fgrp = self.get_fgrp_from_index(curr_index)
            curr_sensor = self.get_sensor_from_index(curr_index)
            if diff_fingers_across_sets and curr_fgrp in anchor_fgrps:
                continue # satisfy (5) - different fingers than anchor, if needed
            if diff_fingers_within_set and curr_fgrp in retVal_fgrps:
                continue # satisfy (6) - different fingers than each other, if needed
            if diff_sensors_across_sets and curr_sensor in anchor_sensors:
                continue # satisfy (7) - different sensors than anchor, if needed
            if same_sensor_within_set and len(retVal_sensors) >= 1 and curr_sensor not in retVal_sensors:
                continue # satisfy (8) - same sensor as each other, if needed
            if curr_fgrp not in possible_fgrps: # TODO: optimize this code???
                continue # satisfy (9) - only use certain fingers
            if self.get_datasetName_from_index(curr_index) != self.get_datasetName_from_index(anchor_indices[0]):
                continue # satisfy (10) - same dataset as anchors
            
            ret_val.append(curr_index) # satisfy (4) - distinct samples than each other
            retVal_fgrps.add(curr_fgrp) # satisfy (6) - different fingers than each other (optionally)
            retVal_sensors.add(curr_sensor) # satisfy (8) - same sensor as each other (optionally)

        # (0) definitely same class as each other
        assert len(set([self.the_labels[x] for x in ret_val])) == 1
        # (1) from same (a) / diff (b) class as anchor_indices, depending on same_class_as_anchor
        assert same_class_as_anchor == (self.the_labels[anchor_indices[0]] == self.the_labels[ret_val[0]])
        # (2) size self.num_pos_fingers (a) or self.num_neg_fingers (b), depending on same_class_as_anchor
        assert (same_class_as_anchor and len(ret_val) == self.num_pos_fingers) \
            or (not same_class_as_anchor and len(ret_val) == self.num_neg_fingers)
        # (3) definitely distinct samples from anchor_indices
        assert len(set(anchor_indices).union(set(ret_val))) == len(anchor_indices) + len(ret_val)
        # (4) definitely distinct samples from each other
        assert len(set(ret_val)) == len(ret_val)
        # (5) (optionally) from different fingers than anchor_indices
        if diff_fingers_across_sets:
            assert len(anchor_fgrps.union(retVal_fgrps)) == len(anchor_fgrps) + len(retVal_fgrps)
        # (6) (optionally) from different fingers than each other
        if diff_fingers_within_set:
            assert len(retVal_fgrps) == len(set(retVal_fgrps))
        # (7) (optionally) from different sensor than anchor_indices
        if diff_sensors_across_sets:
            assert len(anchor_sensors.union(retVal_sensors)) == len(anchor_sensors) + len(retVal_sensors)
        # (8) (optionally) from same sensor as each other
        if same_sensor_within_set:
            assert len(retVal_sensors) == 1
        # (9) only use certain fingers
        assert retVal_fgrps.issubset(set(possible_fgrps))
        # (10) same dataset as anchor
        assert set([self.get_datasetName_from_index(i) for i in ret_val]) == set([self.get_datasetName_from_index(i) for i in anchor_indices])
        
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
    
    def get_datasetName_from_index(self, i):
        return self.get_dataset_name(self.the_data[i])

    def get_filename_from_index(self, i):
        return self.the_data[i].split('/')[-1]
    
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
        if self.train: # randomized
            return self.get_item_train(index)
        else: # deterministic
            return self.get_item_test(index)

    """
    If training, should only be one at a time
    """
    def get_item_train(self, index):
        # TODO: debug this
        anchor_filepath = self.the_data[self.train_anchor_indices[index]]
        anchor_img = my_transformation(my_read_image(anchor_filepath), train=self.train)
        curr_pos_indices = self.get_indices(anchor_indices=[self.train_anchor_indices[index]],\
                        same_class_as_anchor=True, \
                        diff_fingers_across_sets=self.diff_fingers_across_sets,\
                        diff_fingers_within_set=self.diff_fingers_within_set, \
                        diff_sensors_across_sets=self.diff_sensors_across_sets, \
                        same_sensor_within_set=self.same_sensor_within_set, \
                        possible_fgrps=self.acceptable_pos_fgrps)
        assert len(curr_pos_indices) == 1
        pos_filepath = self.the_data[curr_pos_indices[0]]
        pos_img = my_transformation(my_read_image(pos_filepath), train=self.train)
        curr_neg_indices = self.get_indices(anchor_indices=[self.train_anchor_indices[index]],\
                        same_class_as_anchor=False, \
                        diff_fingers_across_sets=self.diff_fingers_across_sets,\
                        diff_fingers_within_set=self.diff_fingers_within_set, \
                        diff_sensors_across_sets=self.diff_sensors_across_sets, \
                        same_sensor_within_set=self.same_sensor_within_set, \
                        possible_fgrps=self.acceptable_neg_fgrps)
        assert len(curr_neg_indices) == 1
        neg_filepath = self.the_data[curr_neg_indices[0]]
        neg_img = my_transformation(my_read_image(neg_filepath), train=self.train)

        curr_iteration_labels = [self.the_labels[self.train_anchor_indices[index]], \
                                    self.the_labels[curr_pos_indices[0]], \
                                    self.the_labels[curr_neg_indices[0]]
        ]
        return (anchor_img, pos_img, neg_img), curr_iteration_labels, (anchor_filepath, pos_filepath, neg_filepath)

    def get_item_test(self, index):
        anchor_filepaths = [self.the_data[i] for i in self.test_triplets[index][0]]
        pos_filepaths = [self.the_data[i] for i in self.test_triplets[index][1]]
        neg_filepaths = [self.the_data[i] for i in self.test_triplets[index][2]]

        the_labels = [self.the_labels[self.test_triplets[index][0][0]], \
                    self.the_labels[self.test_triplets[index][1][0]], \
                    self.the_labels[self.test_triplets[index][2][0]]]
        assert the_labels[0] == the_labels[1]
        assert the_labels[0] != the_labels[2]

        anchor_imgs = [my_transformation(my_read_image(x), train=self.train) for x in anchor_filepaths]
        pos_imgs = [my_transformation(my_read_image(x), train=self.train) for x in pos_filepaths]
        neg_imgs = [my_transformation(my_read_image(x), train=self.train) for x in neg_filepaths]
        
        # for backwards compatability in train loop - remove singleton tuples, just return the items themselves
        if self.num_anchor_fingers == 1 and self.num_pos_fingers == 1 and self.num_neg_fingers == 1:
            return (anchor_imgs[0], pos_imgs[0], neg_imgs[0]), \
                (the_labels[0][0], the_labels[1][0], the_labels[2][0]), \
                (anchor_filepaths[0], pos_filepaths[0], neg_filepaths[0])

        return (anchor_imgs, pos_imgs, neg_imgs), the_labels, (anchor_filepaths, pos_filepaths, neg_filepaths)

    def __len__(self):
        if not self.train: # we can have multiple testing triplets for each item in the dataset
            return len(self.test_triplets)
        return len(self.train_anchor_indices)
        #return len(self.fingerprint_dataset)


