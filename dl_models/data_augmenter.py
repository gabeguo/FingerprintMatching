import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image
import sys
import os
import math

sys.path.append('../')

from trainer import *
from losses import *
from siamese_datasets import *
from fingerprint_dataset import *
from embedding_models import *

from common_filepaths import DATA_FOLDER

MODEL_PATH = '/data/therealgabeguo/embedding_net_weights.pth'

batch_size=1

training_dataset = TripletDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'train'), train=True))
#training_dataset = torch.utils.data.Subset(training_dataset, list(range(0, len(training_dataset), 50)))
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)

# SHOW IMAGES
import matplotlib.pyplot as plt
print("Training Images")
counter = 0
for images, labels, filepaths in train_dataloader:
    image, label, filepath = images[0], labels[0], filepaths[0] # Can do this because batch size = 1
    if counter == 10:
        break
    print("Label is: ", label)
    print("Filepath is: ", filepath)
    print("Shape of image: ", image.size())
    head, tail = os.path.split(filepath)
    filename = tail.split(".")[0]
    augmented_file_path = os.path.join(head, "augmented_images", filename + "_aug1.png")
    save_image(image, augmented_file_path)
    #next_img = image[2][0]
    #the_min = torch.min(next_img)
    #the_max = torch.max(next_img)
    #next_img = (next_img - the_min) / (the_max - the_min)
    #print(filepaths[2][0])
    #print(next_img[0])
    #plt.imshow(next_img.permute(1, 2, 0))
    #plt.show()
    counter += 1
