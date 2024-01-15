import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from torchvision import  transforms


class GetDVS(Dataset):
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
        new_data = []
        for t in range(data.size(0)):
            if self.resize:
                new_data.append(self.tensorx(self._resize(self.imgx(data[t,...]))))
            else:
                new_data.append(torch.tensor(data[t,...]))
        data = torch.stack(new_data, dim=0)
        if self.transform:
            off1 = random.randint(-25, 25)
            off2 = random.randint(-25, 25)
            data = transforms.functional.affine(data, angle = 0.0, scale=1.0, shear=0.0, translate=(off1, off2))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))