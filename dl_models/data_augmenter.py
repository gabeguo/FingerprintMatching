import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
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
for image, label, filepath in train_dataloader:
    if counter == 10:
        break
    print("Label is: ", label)
    print("Filepath is: ", filepath)
    print("Shape of image: ", image.size())
    #next_img = image[2][0]
    #the_min = torch.min(next_img)
    #the_max = torch.max(next_img)
    #next_img = (next_img - the_min) / (the_max - the_min)
    #print(filepaths[2][0])
    #print(next_img[0])
    #plt.imshow(next_img.permute(1, 2, 0))
    #plt.show()
    counter += 1
