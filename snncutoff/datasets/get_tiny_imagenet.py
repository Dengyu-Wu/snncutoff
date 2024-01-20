import torch
import torchvision.transforms as transforms
import os
from torchvision import datasets, transforms
from snncutoff.augmentation import ImageNetPolicy


def GetTinyImageNet(dataset_path, attack=False):
    trans_t = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ])
    
    trans = transforms.Compose([
                            transforms.ToTensor(), 
                            ])

    train_data = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=trans_t)
    test_data = datasets.ImageFolder(root=os.path.join(dataset_path, 'val'), transform=trans)
    return train_data, test_data
