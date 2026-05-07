import sys

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import transforms
from pycocotools.coco import COCO

import torchvision.datasets as datasets
"""
def get_data(slice=1, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  equiv to slicing with [::slice] 
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader


def make(config, device="cuda"):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer


"""

def get_data(slice=1, train=True):

    if train:
        root = "/home/datasets/coco/train2017"
        annFile = "/home/datasets/coco/annotations/instances_train2017.json"
    else:
        root = "/home/datasets/coco/val2017"
        annFile = "/home/datasets/coco/annotations/instances_val2017.json"

    coco = COCO(annFile)

    # carregar dataset base
    full_dataset = datasets.CocoDetection(
        root=root,
        annFile=annFile,
        transform=transforms.ToTensor()
    )

    person_id = 1  # COCO = person

    def make_sample(idx):
        image, anns = full_dataset[idx]

        h, w = image.shape[1], image.shape[2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in anns:
            if ann["category_id"] == person_id:
                mask = np.maximum(mask, coco.annToMask(ann))

        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask

    # wrapper dataset simple
    class Wrapped(torch.utils.data.Dataset):
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return make_sample(idx)

    dataset = Wrapped(len(full_dataset))

    # subsampling com el teu MNIST
    sub_dataset = Subset(
        dataset,
        indices=range(0, len(dataset), slice)
    )

    return sub_dataset

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader


dataset = get_data(slice=1, train=True)

print(len(dataset))

image, mask = dataset[0]

print(image.shape)  # (3, H, W)
print(mask.shape)   # (H, W)
print(mask.unique())  # [0,1]