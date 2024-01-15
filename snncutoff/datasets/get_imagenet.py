import torch
import torchvision.transforms as transforms
import os
from torchvision import datasets, transforms
from snncutoff.augmentation import ImageNetPolicy


def GetImageNet(dataset_path, attack=False):
    trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                # ImageNetPolicy(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    
    trans = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    train_data = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=trans_t)
    test_data = datasets.ImageFolder(root=os.path.join(dataset_path, 'val'), transform=trans)
    return train_data, test_data
