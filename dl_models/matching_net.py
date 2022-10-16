import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MatchingNet(nn.Module):
    def __init__(self, embedding_len = 1000):
        self.feature_extractor = models.resnet18(pretrained=True)
        num_features = self.feature_extractor.in_features
        self.feature_extractor.fc = nn.Linear(num_features, embedding_len)
        self.distance = torch.cdist
        self.final_activation = nn.Sigmoid()
        return

    def forward(self, x1, x2):
        output1 = self.feature_extractor(x1)
        output2 = self.feature_extractor(x2)
        distance = self.distance(x1, x2, p=2)
        prob_same_class = self.final_activation(distance)
        return prob_same_class
