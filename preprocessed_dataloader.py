# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries for data manipulation and visualization
import os
from pathlib import Path
import numpy as np


class PreprocessedDataset(Dataset):

    def __init__(self, image_dir=(str(Path.home()) + "/processed"), device='cpu'):
        self.image_dir = image_dir
        self.device = device

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, "/image-" + str(item))

        image, label = torch.load(image_path, self.device)

        return (image, label)


def processed_split_loaders(no_folds, fold,  batch_size, seed, device, p_test=0.1, shuffle=True,
                         extras={}):
    """ Modified from create_split_loaders in xray_dataloader.py by Jenny Hamer.

    Params:
    -------
    - batch_size: (int) mini-batch size to load at a time
    - seed: (int) Seed for random generator (use for testing/reproducibility)
    - transform: A torchvision.transforms object - transformations to apply to each image
                 (Can be "transforms.Compose([transforms])")
    - p_val: (float) Percent (as decimal) of dataset to use for validation
    - p_test: (float) Percent (as decimal) of the dataset to split for testing
    - shuffle: (bool) Indicate whether to shuffle the dataset before splitting
    - extras: (dict)
        If CUDA/GPU computing is supported, contains:
        - num_workers: (int) Number of subprocesses to use while loading the dataset
        - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory
                  (set to True if using a GPU)
        Otherwise, extras is an empty dict.

    Returns:
    --------
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    - test_loader: (DataLoader) The iterator for the test set
    """

    # Get create a ChestXrayDataset object
    dataset = PreprocessedDataset(device=device)  # TODO add directory?

    # Dimensions and indices of training set
    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)

    # set the proportion of division of the training data
    portion = float(fold) / float(no_folds) + 1.0

    # Separate a test split from the training dataset
    test_split = int(np.floor(p_test * len(all_indices)))
    train_ind, test_ind = all_indices[test_split:], all_indices[: test_split]

    # TODO verify
    train_set = set(train_ind)
    val_ind = set(train_ind[int((fold-1) * portion): int(fold * portion)])
    train_ind = train_set.difference(val_ind)

    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = SubsetRandomSampler(train_ind)
    sample_test = SubsetRandomSampler(test_ind)
    sample_val = SubsetRandomSampler(val_ind)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=sample_train, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=sample_test, num_workers=num_workers,
                             pin_memory=pin_memory)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers,
                            pin_memory=pin_memory)

    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)


