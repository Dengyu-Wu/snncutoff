import argparse
import shutil
import os
import time
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from easycutoff.models.resnet_models import resnet19
from easycutoff.models.vggsnns import *
from easycutoff import data_loaders
from easycutoff.functions import TET_loss
import numpy as np
from configs import *
from omegaconf import DictConfig, OmegaConf
import hydra
from easycutoff.utils import replace_activation_by_neuron, ann_to_snn_conversion, reset_neuron
from typing import Callable, List, Type
from .cutoff import BaseCutoff
from easycutoff.utils import  OutputHook, sethook

class SNNCASE:
    def __init__(
        self,
        method: str,
        criterion: nn.Module,
        args: dict
    ) -> None:
        """A unified, easy-to-use API for SNN training
        """
        self.method = method
        self.criterion = criterion
        self.args = args
    def preprocess(self,x):
        if self.method == 'ann':             
            return x.sum(1)
        elif self.method == 'snn':
            x = x.transpose(0,1)
            return x 
        else:
            ValueError('Wrong training method')

    def postprocess(self, x, y):
        if self.args.TET:
            T = x.size()[0]
            Loss_es = 0

            for t in range(T):
                Loss_es += self.criterion(x[t, ...], y)
            Loss_es = Loss_es / T # L_TET  
            if self.args.lamb != 0:
                MMDLoss = torch.nn.MSELoss()
                y = torch.zeros_like(x).fill_(self.args.means)
                Loss_mmd = MMDLoss(x, y) # L_mse
            else:
                Loss_mmd = 0
            return x.mean(0), (1 - self.args.lamb) * Loss_es + self.args.lamb * Loss_mmd # L_Total
        else:
            if self.args.multistep:
                x = x.mean(0)
            loss = self.criterion(x,y)
            return x,loss

    
