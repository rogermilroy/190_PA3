import torch
import torchvision.transforms as transforms
import xray_dataloader
import os
from pathlib import Path
import concurrent.futures


def save_image(int, data):
    image_label = data[int]
    torch.save(image_label, str(Path.home()) + '/processed/image-' + str(i))


if not os.path.exists(str(Path.home()) + '/processed'):
    os.makedirs(str(Path.home()) + '/processed')

transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])

dataset = xray_dataloader.ChestXrayDataset(transform=transform)

for i in range(len(dataset)):
    save_image(i, dataset)



