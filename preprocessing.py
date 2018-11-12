import torch
import torchvision.transforms as transforms
import xray_dataloader
import preprocessed_dataloader
import os
from pathlib import Path

def process():
    def save_image(int, data):
        image_label = data[int]
        torch.save(image_label, str(Path.home()) + '/processed/image-' + str(int))


    if not os.path.exists(str(Path.home()) + '/processed'):
        os.makedirs(str(Path.home()) + '/processed')

    transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])

    dataset = xray_dataloader.ChestXrayDataset(transform=transform)

    for i in range(len(dataset)):
        save_image(i, dataset)

def augment_flip():

    dataset = preprocessed_dataloader.PreprocessedDataset()

    tran = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(1.0),
                               transforms.ToTensor()])
    index = 112120
    filename = str(Path.home()) + '/processed/image-'

    for i in range(len(dataset)):
        image, label = dataset[i]
        # don't flip cardiomegaly images because the heart doesn't change orientations (almost ever)
        if label[1] == 1.0:
            continue
        elif torch.sum(label).item() >= 1.0:
            flipped_image = tran(image)
            to_store = (flipped_image, label)
            torch.save(to_store, filename + str(index))
            index += 1


def create_subset():
    train, val, subset = preprocessed_dataloader.processed_split_loaders(1, 0, 128, 12)
    i = 0
    for minibatch_count, image_label in enumerate(subset, 0):
        for image in image_label:
            torch.save(image, str(Path.home()) + '/processed/subset/image-' + str(i))
            i += 1


if __name__ == '__main__':
    create_subset()



