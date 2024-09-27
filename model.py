import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

# customized model

def get_custom_ResNet(input_channels, num_classes):
    """ResNet18"""
    class Residual(nn.Module):
        def __init__(self, input_channels, num_channels, strides=1, use_1x1conv=False):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, num_channels, 
                                   kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.Conv2d(num_channels, num_channels,
                                   kernel_size=3, padding=1)
            if use_1x1conv:
                self.res_conv = nn.Conv2d(input_channels, num_channels,
                                         kernel_size=1, stride=strides)
            
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)
        
        def forward(self, X):
            Y = F.relu(self.bn1(self.conv1(X)))
            Y = self.bn2(self.conv2(Y))
            if hasattr(self, 'res_conv'):
                X = self.res_conv(X)
            Y += X
            return F.relu(Y)
    
    def resnet_block(input_channels, num_channels,
                     num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and first_block:
                blk.append(Residual(input_channels, num_channels,
                                    strides=2, use_1x1conv=True))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk
    
    # ResNet18
    b1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True), nn.Dropout())
    b3 = nn.Sequential(*resnet_block(64, 128, 2), nn.Dropout())
    b4 = nn.Sequential(*resnet_block(128, 256, 2), nn.Dropout())
    b5 = nn.Sequential(*resnet_block(256, 512, 2), nn.Dropout())

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(512, num_classes))
    
    return net

# pretrained models

def get_model_ResNet18(num_classes):
    net = models.resnet18(pretrained=True)
    net.fc = nn.Sequential(nn.Linear(net.fc.in_features, num_classes))
    return net

def get_model_ResNet50(num_classes):
    net = models.resnet50(pretrained=True)
    net.fc = nn.Sequential(nn.Linear(net.fc.in_features, num_classes))
    return net