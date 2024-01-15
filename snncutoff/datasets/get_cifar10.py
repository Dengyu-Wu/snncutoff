import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from snncutoff.augmentation import Cutout, CIFAR10Policy

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