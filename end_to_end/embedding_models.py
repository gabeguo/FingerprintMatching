# Thanks https://github.com/adambielski/siamese-triplet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EmbeddingNet(nn.Module):
    def __init__(self, pretrained=True):
        super(EmbeddingNet, self).__init__()
        # Thanks # https://discuss.pytorch.org/t/how-can-l-use-the-pre-trained-resnet-to-extract-feautres-from-my-own-dataset/9008
        # self.the_resnet = models.resnext50_32x4d(pretrained=pretrained)
        self.the_resnet = models.resnet18(pretrained=pretrained)
        # self.the_resnet = models.resnet34(pretrained=pretrained)
        modules=list(self.the_resnet.children())[:-1]
        self.feature_extractor=nn.Sequential(*modules)
        self.fc = self.the_resnet.fc

    def forward(self, x):
        output = F.normalize(self.feature_extractor(x), p=2, dim=1)
        #print(output.shape)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class EndtoEndNet(nn.Module):
    def __init__(self, embedding_net):
        super(EndtoEndNet, self).__init__()
        self.embedding_net = embedding_net
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        #print(output1.size())
        #print(torch.squeeze(torch.cat((output1, output2), 1)).size())
        logit = self.head(torch.squeeze(torch.cat((output1, output2), 1)))
        return logit
    
    def get_embedding(self, x):
        return self.embedding_net(x)#