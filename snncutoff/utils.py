import torch
import torch.nn as nn
from .modules import TCL, MyFloor, ScaledNeuron, StraightThrough
import logging
import random
import os
import numpy as np
from snncutoff.neuron import *
from snncutoff.ann_constrs import PreConstrs, PostConstrs
from snncutoff.snn_layers import BaseLayer
from snncutoff.snn_layers import TEBN

def seed_all(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False

def isContainer(name):
    if 'container' in name.lower():
        return True
    return False

def addPreConstrs(name):
    if 'conv2d' == name.lower() or 'linear' == name.lower() or 'pool' in name.lower() or 'flatten' in name.lower():
        return True
    return False

def addPostConstrs(name):
    if 'pool' in name.lower() or 'flatten' in name.lower():
        return True
    return False

def addSingleStep(name):
    if  'lifspike' in name:
        return True
    if 'constrs' in name or 'baselayer' in name:
        if  'preconstrs' in name or 'postconstrs' in name:
            return False
        else:
            return True
    return False

def multi_to_single_step(model,reset_mode):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = multi_to_single_step(module,reset_mode)
        if addSingleStep(module.__class__.__name__.lower()):
            model._modules[name] = BaseLayer(vthr=model._modules[name].vthr, 
                                             tau=model._modules[name].tau, 
                                             multistep=False, 
                                             reset_mode=reset_mode)
        if  'preconstrs' in module.__class__.__name__.lower():
            model._modules[name].multistep=False  
        # if  'dropout' in module.__class__.__name__.lower():
        #     model._modules[name] = LinearConstrs(T=1)
        if  'postconstrs' in module.__class__.__name__.lower():
            model._modules[name].multistep=False  
    return model

def _add_ann_constraints(model, T, L, ann_constrs, regularizer=None):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = _add_ann_constraints(module, T, L, ann_constrs,regularizer)
        if  'relu' == module.__class__.__name__.lower():
            model._modules[name] = ann_constrs(T=T, L=L, regularizer=regularizer)
        if  addPreConstrs(module.__class__.__name__.lower()):
            model._modules[name] = PreConstrs(T=T, module=model._modules[name])
        if  addPostConstrs(module.__class__.__name__.lower()):
            model._modules[name] = PostConstrs(T=T, module=model._modules[name])    
    return model

def add_ann_constraints(model, T, L, ann_constrs, regularizer=None):
    model = _add_ann_constraints(model, T, L, ann_constrs, regularizer=regularizer)
    model = nn.Sequential(
        *list(model.children()),  
        PostConstrs(T=T, module=None)    # Add the new layer
        )
    return model

def addSNNLayers(name):
    if 'relu' == name.lower():
        return True
    return False


def _add_snn_layers(model, T, snn_layers, regularizer=None, TBN=None):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = _add_snn_layers(module, T, snn_layers,regularizer, TBN=TBN)
        if  addSNNLayers(module.__class__.__name__.lower()):
            model._modules[name] = snn_layers(T=T, regularizer=regularizer)
        if  addPreConstrs(module.__class__.__name__.lower()):
            model._modules[name] = PreConstrs(T=T, module=model._modules[name])
        if  addPostConstrs(module.__class__.__name__.lower()):
            model._modules[name] = PostConstrs(T=T, module=model._modules[name])    
        if TBN:
            if  'norm2d' in module.__class__.__name__.lower():
                model._modules[name] = TEBN(T=T, num_features=model._modules[name].num_features)  
    return model

def add_snn_layers(model, T, snn_layers, TBN=False, regularizer=None):
    model = _add_snn_layers(model, T, snn_layers, regularizer=regularizer,TBN=TBN)
    model = nn.Sequential(
        *list(model.children()),  
        PostConstrs(T=T, module=None)    # Add the new layer
        ) 
    return model

def reset_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = reset_neuron(module)
        if hasattr(module, "neuron"):
            model._modules[name].neuron.reset()
    return model


def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model

class OutputHook(list):
    def __init__(self):
        self.mask = 0                
    def __call__(self, module, inputs, output):
        loss = output
        self.append(loss)      
                

class sethook(object):
    def __init__(self,output_hook):
        self.module_dict = {}
        self.k = 0
        self.output_hook = output_hook
    def get_module(self,model):
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                model._modules[name] = self.get_module(module)
            if hasattr(module, "add_loss"):
                self.module_dict[str(self.k)] = module
                self.k+=1
        return model
    
    def __call__(self,model,remove=False):
        model = self.get_module(model)
        if remove:
            self.remove()
        else:
            self.set_hook()
        return model                       
    def set_hook(self):
        for y,x in self.module_dict.items():
            self.remove_all_hooks(self.module_dict[y])
            self.module_dict[y] = self.module_dict[y].register_forward_hook(self.output_hook) 
            
    def remove_all_hooks(self,module):
        from collections import OrderedDict
        child = module
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict()
    def remove(self):
        for y,x in self.module_dict.items():
            self.remove_all_hooks(self.module_dict[y])

import pickle
import os

def save_pickle(mdl, name, path):
    pklout = open( os.path.join(path, name + '.pkl'), 'wb')
    pickle.dump(mdl, pklout)
    pklout.close()
    
def load_pickle(path):
    pkl_file = open(path, 'rb')
    mdl = pickle.load(pkl_file)
    pkl_file.close()

    return mdl