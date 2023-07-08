import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils
from torchvision.io import read_image, ImageReadMode
import numpy as np
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
        transforms.Normalize([0, 0, 0], [1, 1, 1]),
    ])
    """
    if train:
        transform = transforms.Compose([
            #transforms.RandomRotation(15, fill=255),
            transform,
        ])
    """
    return transform(the_image.float())

class FingerprintDataset(Dataset):
    def is_image_filename(self, filename):
        return any(extension in filename for extension in ['.png', \
            '.PNG', '.pneg', '.PNEG', '.jpg', '.JPG', '.jpeg', '.JPEG'])

    """
    Loads the unpaired images from the folder
    1) Stores indexed unpaired images with their corresponding labels in self.images, self.img_labels
    2) Stores all the possible classes in self.classes
    3) Stores images separated by class in self.class_to_images (has image filepath, not actual image)
    """
    def load_images(self, root_dir):
        self.classes = list() # self.classes[i] = the name of class with index i
        self.img_labels = list()
        self.images = list()
        self.class_to_images = list()

        for pid in os.listdir(root_dir):
            self.classes.append(pid)
            self.class_to_images.append(list())
            curr_person_folder = os.path.join(root_dir, pid)
            for sample in os.listdir(curr_person_folder):
                if not self.is_image_filename(sample):
                    continue
                curr_image = os.path.join(curr_person_folder, sample)

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
        else:
            self.test_labels = self.img_labels
            self.test_data = self.images

        return

    def __len__(self):
        return self.len

    # returns image, label, filepath
    def __getitem__(self, idx):
        return my_transformation(read_image(self.images[idx], mode=ImageReadMode.RGB), train=self.train), \
        self.img_labels[idx], \
        self.images[idx]
