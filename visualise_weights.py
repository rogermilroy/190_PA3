import torch
import torchvision
import matplotlib.pyplot as plt
from deep_and_small import DeepAndSmallCNN
from pathlib import Path

image = torch.load(str(Path.home()) + '/image-20')

model = DeepAndSmallCNN()

weights = torch.load('./results/kfold-deep-small/test-0-params')

model.load_state_dict(weights)

output = model(image)

