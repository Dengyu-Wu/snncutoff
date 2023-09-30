import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST
import warnings
import os
import torchvision
from os import listdir
import numpy as np
import time
from os.path import isfile, join

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
                new_data.append(self.tensorx(self.imgx(data[t,...])))
        data = torch.stack(new_data, dim=0)
        if self.transform is not None:
            # flip = random.random() > 0.5
            # if flip:
            #     data = torch.flip(data, dims=(3,))
            off1 = random.randint(-26, 26)
            off2 = random.randint(-26, 26)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


def get_dvs_loaders(path, batch_size=32, workers=10, resize=False):
    train_path = path + '/train'
    val_path = path + '/test'
    train_dataset = DVS_Dataset(root=train_path, transform=True, resize=resize)
    val_dataset = DVS_Dataset(root=val_path, resize=resize)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                             num_workers=workers, pin_memory=True)
    # test_loader = DataLoader(val_dataset, batch_size=batch_size,
    #                             shuffle=False, num_workers=workers, pin_memory=True)

    return train_dataset, val_dataset

if __name__ == '__main__':
    train_set, test_set = get_dvs_loaders()
