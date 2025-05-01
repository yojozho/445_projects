"""
EECS 445 - Introduction to Machine Learning
Winter 2024 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Challenge(nn.Module):
    def __init__(self):
        """
        Define the architecture, i.e. what layers our network contains.
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions.
        """
        super().__init__()

        ## TODO: define your model architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.fc_1 = nn.Linear(in_features=32, out_features=2)
        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        ## TODO: initialize the parameters for your network
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc_1]
        F_in = self.fc_1.weight.size(1)
        nn.init.normal_(self.fc_1.weight, 0.0, 1/sqrt(F_in))
        nn.init.constant_(self.fc_1.bias, 0.0)

    def forward(self, x):
        """
        This function defines the forward propagation for a batch of input examples, by
        successively passing output of the previous layer as the input into the next layer (after applying
        activation functions), and returning the final output as a torch.Tensor object.

        You may optionally use the x.shape variables below to resize/view the size of
        the input matrix at different points of the forward pass.
        """
        N, C, H, W = x.shape

        ## TODO: implement forward pass for your network
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F. relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        flatten = torch.nn.Flatten()
        x = flatten(x)
        x = self.fc_1(x)
        return x
