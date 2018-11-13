import torch
import torchvision
import matplotlib.pyplot as plt
from deep_and_small import DeepAndSmallCNN
from baseline_cnn import BasicCNN
import testing
import os

#image = torch.load(os.getcwd() + '/images/image-20')

#model = DeepAndSmallCNN()

#weights = torch.load(os.getcwd() + '/results/kfold-deep-small/test-0-params')

#model.load_state_dict(weights)

#output = model(image)

# Saves content of directory into list
path = './results/kfold-basic-weighted-1'
files = os.listdir(path)

# Filters into aggregate vs epoch
aggregatefiles = []
epochfiles = []

for name in files:
    if('val-loss-0-conf-' in name):
        epochfiles.append(name)
    else:
        aggregatefiles.append(name)

losses = 0
per_class = 0
aggregated = 0
conf = 0
for filename in aggregatefiles:
    if('agg' in filename):
        aggregated =
    if('loss' in filename):
        d
    if('class' in filename):

