import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import warnings
import os
import torchvision
from os import listdir
import numpy as np
import time
from os.path import isfile, join
from textwrap import fill
from torchvision import datasets, transforms
from easycutoff.preprocessing.utils import Cutout, CIFAR10Policy

warnings.filterwarnings('ignore')

class DVS_Dataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, resize=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self._resize = transforms.Resize(size=(48, 48))  
        self.resize = resize
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()
        self.affine = transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        # print(data.shape)
        # if self.train:
        new_data = []
        for t in range(data.size(0)):
            if self.resize:
                new_data.append(self.tensorx(self._resize(self.imgx(data[t,...]))))
            else:
                # new_data.append(self.tensorx(self.imgx(data[t,...])))
                new_data.append(torch.tensor(data[t,...]))
        data = torch.stack(new_data, dim=0)
        if not self.transform:
            # flip = random.random() > 0.5
            # if flip:
            #     data = torch.flip(data, dims=(3,))
            off1 = random.uniform(0, 0.2)
            off2 = random.uniform(0, 0.2)
            data = transforms.functional.affine(data, angle = 0.0, scale=1.0, shear=0.0, translate=(off1, off2))
            # data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


# def get_dvs_loaders(path, data='cifar10-dvs', resize=False):
#     train_path = path + '/train'
#     val_path = path + '/test'
#     train_dataset = DVS_Dataset(root=train_path, transform=True, resize=resize)
#     val_dataset = DVS_Dataset(root=val_path, resize=resize)

#     return train_dataset, val_dataset

def isDVSData(name):
    if 'cifar10-dvs' in name.lower() or 'ncaltech101' in name.lower() or 'dvs128-gesture' in name.lower():
        return True
    return False

def get_data_loaders(path, data, resize=False):
    if data.lower() == 'cifar10':
        return GetCifar10(path)
    elif data.lower() == 'cifar100':
        return GetCifar100(path)
    elif data.lower() == 'imagenet':
        return GetImageNet(path)
    elif isDVSData(data):
        train_path = path + '/train'
        val_path = path + '/test'
        train_dataset = DVS_Dataset(root=train_path, transform=True, resize=resize)
        val_dataset = DVS_Dataset(root=val_path, resize=resize)
        return train_dataset, val_dataset
    else:
        NameError("The dataset name is not support!")
        exit(0)

def GetCifar10(dataset_path, attack=False):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    if attack:
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10(dataset_path, train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(dataset_path, train=False, transform=trans, download=True) 
    return train_data, test_data


def GetCifar100(dataset_path, attack=False):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  CIFAR10Policy(),
                                  transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])
    train_data = datasets.CIFAR100(dataset_path, train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(dataset_path, train=False, transform=trans, download=True) 
    return train_data, test_data

def GetImageNet(dataset_path, attack=False):
    trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
