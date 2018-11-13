import torch
import torchvision
import matplotlib.pyplot as plt
from deep_and_small import DeepAndSmallCNN
from baseline_cnn import BasicCNN
import testing
import os

image = torch.load(os.getcwd() + '/images/image-20')

model = DeepAndSmallCNN()

#weights = torch.load(os.getcwd() + '/results/kfold-deep-small/test-0-params')

#model.load_state_dict(weights)

#output = model(image[0])
img = image[0]
plt.gray()
plt.imshow(img[0])
print(img[0].shape)
plt.imshow(img[0])

print(image)
#print(output)
