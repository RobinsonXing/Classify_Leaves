import torch
from torch import nn
from torchvision import models

class ResNetClassifier(nn.Module):

    def __init__(self, num_calsses):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrain=True)
        self.resnet.fc = nn.Linear(512, num_calsses)
    
    def forward(self, x):
        return self.resnet(x)