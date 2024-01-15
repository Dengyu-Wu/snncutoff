from .datasets import *
import warnings 

warnings.filterwarnings('ignore')

def isDVSData(name):
    if 'cifar10-dvs' in name.lower() or 'ncaltech101' in name.lower() or 'dvs128-gesture' in name.lower():
        return True
    return False

def get_data_loaders(path, data,transform=True, resize=False):
    if data.lower() == 'cifar10':
        return GetCifar10(path)
    elif data.lower() == 'cifar100':
        return GetCifar100(path)
    elif data.lower() == 'imagenet':
        return GetImageNet(path)
    elif isDVSData(data):
        train_path = path + '/train'
        val_path = path + '/test'
        train_dataset = GetDVS(root=train_path, transform=transform, resize=resize)
        val_dataset = GetDVS(root=val_path, resize=resize)
        return train_dataset, val_dataset
    else:
        NameError("The dataset name is not support!")
        exit(0)

