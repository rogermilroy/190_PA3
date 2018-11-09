import preprocessed_dataloader
import torch


def main():
    full_dataset = preprocessed_dataloader.PreprocessedDataset()

    freq = None
    for i in range(len(full_dataset)):
        image, label = full_dataset[i]
        if i == 0:
            freq = torch.zeros_like(label)
        freq += label

    print(freq)


if __name__ == '__main__':
    main()
