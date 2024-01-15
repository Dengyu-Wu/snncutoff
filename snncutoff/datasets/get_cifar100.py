
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from snncutoff.augmentation import Cutout, CIFAR10Policy


def GetCifar100(dataset_path, attack=False):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])
    train_data = datasets.CIFAR100(dataset_path, train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(dataset_path, train=False, transform=trans, download=True) 
    return train_data, test_data