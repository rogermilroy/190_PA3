################################################################################
# CSE 190: Programming Assignment 3
# Fall 2018
# Code author: Jenny Hamer modified by Roger Milroy.
#
# Filename: deep_cnn.py
#
# Description:
#
# This file contains the starter code for the baseline architecture you will use
# to get a little practice with PyTorch and compare the results of with your
# improved architecture.
#
################################################################################


# PyTorch and neural network imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
from xray_dataloader import ChestXrayDataset, create_split_loaders

import matplotlib.pyplot as plt
import numpy as np
import os


class DeepAndSmallCNN(nn.Module):
    """ A basic convolutional neural network model for baseline comparison.

    Consists of eight Conv2d layers, two max-pooling layers,
    and 1 fully-connected (FC) layer:

    conv1 -> conv2 -> conv3 -> conv4 -> conv5 -> conv6 -> maxpool -> fc1 -> (outputs)

    """

    def __init__(self):
        super(DeepAndSmallCNN, self).__init__()

        # conv1: 1 input channel, 12 output channels, [8x8] kernel size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        # Add batch-normalization to the outputs of conv1
        self.conv1_normed = nn.BatchNorm2d(16)

        # Initialized weights using the Xavier-Normal method
        torch_init.xavier_normal_(self.conv1.weight)

        # conv2: 24 input channels, 16 output channels, [8x8] kernel
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=18, kernel_size=3, padding=1)
        self.conv2_normed = nn.BatchNorm2d(18)
        torch_init.xavier_normal_(self.conv2.weight)

        # conv3: X input channels, 8 output channels, [6x6] kernel
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=16, kernel_size=7, stride=2)
        self.conv3_normed = nn.BatchNorm2d(16)
        torch_init.xavier_normal_(self.conv3.weight)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv4_normed = nn.BatchNorm2d(16)
        torch_init.xavier_normal_(self.conv4.weight)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=14, kernel_size=3, padding=1)
        self.conv5_normed = nn.BatchNorm2d(14)
        torch_init.xavier_normal_(self.conv5.weight)

        self.conv6 = nn.Conv2d(in_channels=14, out_channels=12, kernel_size=3, padding=1)
        self.conv6_normed = nn.BatchNorm2d(12)
        torch_init.xavier_normal_(self.conv6.weight)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

        self.conv7 = nn.Conv2d(in_channels=12, out_channels=10, kernel_size=3)
        self.conv7_normed = nn.BatchNorm2d(10)
        torch_init.xavier_normal_(self.conv7.weight)

        self.conv8 = nn.Conv2d(in_channels=10, out_channels=8, kernel_size=3)
        self.conv8_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv8.weight)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(in_features=12800, out_features=14)
        torch_init.xavier_normal_(self.fc.weight)

    def forward(self, batch):
        """Pass the batch of images through each layer of the network, applying
        non-linearities after each layer.

        Note that this function *needs* to be called "forward" for PyTorch to
        automagically perform the forward pass.

        Params:
        -------
        - batch: (Tensor) An input batch of images

        Returns:
        --------
        - logits: (Variable) The output of the network
        """

        # Apply first convolution, followed by ReLU non-linearity;
        # use batch-normalization on its outputs
        # with torch.no_grad():
        batch = func.rrelu(self.conv1_normed(self.conv1(batch)))

        # Apply conv2 and conv3 similarly
        batch = func.prelu(self.conv2_normed(self.conv2(batch)))

        batch = func.rrelu(self.conv3_normed(self.conv3(batch)))

        # Pass the output of conv3 to the pooling layer

        batch = func.rrelu(self.conv4_normed(self.conv4(batch)))

        batch = func.rrelu(self.conv5_normed(self.conv5(batch)))

        batch = func.rrelu(self.conv6_normed(self.conv6(batch)))

        batch = self.pool(batch)

        batch = func.rrelu(self.conv7_normed(self.conv7(batch)))

        batch = func.rrelu(self.conv8_normed(self.conv8(batch)))

        batch = self.pool2(batch)

        # Reshape the output of the conv3 to pass to fully-connected layer
        batch = batch.view(-1, self.num_flat_features(batch))

        # Connect fc1 to fc2 - this layer is slightly different than the rest (why?) no relu.
        batch = self.fc(batch)

        # Return the class predictions
        return func.sigmoid(batch)

    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features
