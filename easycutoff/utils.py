import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from .modules import TCL, MyFloor, ScaledNeuron, StraightThrough
import logging
import random
import os
import numpy as np
from easycutoff.neuron import *
from easycutoff.ann_constrs import PreConstrs, PostConstrs


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

def ann_to_snn_conversion(model,layers):
    for child in model.children():
        if hasattr(child,"children"):
            model,layers = ann_to_snn_conversion(child,layers)
                # if isActivation(module.__class__.__name__.lower()):
        # else:
        #     print(child.__class__.__name__.lower())
        if isContainer(child.__class__.__name__.lower()):
            layers.append(child)
        if  'clip' in child.__class__.__name__.lower() or 'temprelu' in child.__class__.__name__.lower():
            # print(child.__class__.__name__.lower())
            if hasattr(child,"moving_max"):
                # layers.append(nn.ReLU())
                # layers.append(child)
                layers.append(IFNeuron(vthr=child.moving_max.item()))
        # if 'vggann' in child.__class__.__name__.lower():
        #     layers.append(child)
        # if 'clip' in child.__class__.__name__.lower():
        #     layers.append(child)

        if 'flatten' in child.__class__.__name__.lower():
            layers.append(child)
        if 'normlayer' in child.__class__.__name__.lower():
            layers.append(child)
    return model,layers

def multi_to_single_step(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = multi_to_single_step(module)
        if  'lifspike' in module.__class__.__name__.lower() or 'constrs' in module.__class__.__name__.lower() :
            model._modules[name] = IFNeuron(vthr=module.thresh)
            # model._modules[name] = IFNeuron(vthr=module.thresh,tau=module.tau)
    return model



def _add_ann_constraints(model, T, ann_constrs, regularizer=None):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = _add_ann_constraints(module, T, ann_constrs,regularizer)
        if  'relu' == module.__class__.__name__.lower():
            model._modules[name] = ann_constrs(T=T, regularizer=regularizer)
        if  addPreConstrs(module.__class__.__name__.lower()):
            model._modules[name] = PreConstrs(T=T, module=model._modules[name])
        if  addPostConstrs(module.__class__.__name__.lower()):
            model._modules[name] = PostConstrs(T=T, module=model._modules[name])    
    return model

def add_ann_constraints(model, T, ann_constrs, regularizer=None):
    model = _add_ann_constraints(model, T, ann_constrs, regularizer=regularizer)
    model = nn.Sequential(
        *list(model.children()),  
        PostConstrs(T=T, module=None)    # Add the new layer
        )
    return model



# def multi_to_single_step(model,layers):
#     for child in model.children():
#         if hasattr(child,"children"):
#             model,layers = multi_to_single_step(child,layers)
#                 # if isActivation(module.__class__.__name__.lower()):
#         # else:
#         #     print(child.__class__.__name__.lower())
#         if isContainer(child.__class__.__name__.lower()):
#             layers.append(child)
#         if  'lifspike' in child.__class__.__name__.lower():
#             # print(child.__class__.__name__.lower())
#             # if hasattr(child,"moving_max"):
#                 # layers.append(nn.ReLU())
#                 # layers.append(child)
#             layers.append(LIFNeuronReg(vthr=child.thresh.item(),tau=child.tau.item()))
#         # if 'vggann' in child.__class__.__name__.lower():
#         #     layers.append(child)
#         # if 'clip' in child.__class__.__name__.lower():
#         #     layers.append(child)

#         if 'flatten' in child.__class__.__name__.lower():
#             layers.append(child)
#         if 'normlayer' in child.__class__.__name__.lower():
#             layers.append(child)
#     return model,layers

def reset_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = reset_neuron(module)
        if 'neuron' in module.__class__.__name__.lower():
                model._modules[name].reset()
    return model

def replace_activation_by_module(model, m):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_module(module, m)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                model._modules[name] = m(module.up.item())
            else:
                model._modules[name] = m()
    return model




def replace_activation_by_floor(model, t):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    model._modules[name] = MyFloor(module.up.item(), t)
            else:
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    model._modules[name] = MyFloor(8., t)
    return model

def replace_activation_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_activation_by_neuron(module)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                model._modules[name] = ScaledNeuron(scale=module.up.item())
            else:
                model._modules[name] = ScaledNeuron(scale=1.)
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

def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model

def _fold_bn(conv_module, bn_module, avg=False):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module, avg=False):
    w, b = _fold_bn(conv_module, bn_module, avg)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2

def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)

def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # set the bn module to straight through
            setattr(model, n, StraightThrough())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


def regular_set(model, paras=([],[],[])):
    for n, module in model._modules.items():
        if isActivation(module.__class__.__name__.lower()) and hasattr(module, "up"):
            for name, para in module.named_parameters():
                paras[0].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
    return paras


# class OutputHook(list):
#     def __init__(self):
#         self.mask = 0                
#     def __call__(self, module, inputs, output):
#         import numpy as np
#         x = output
#         rank = len(x.size())-1   # N,T,C,W,H
#         rank = np.clip(rank,3,rank)
#         dim = np.arange(rank-1)+2   
#         dim = list(dim)

        
#         x_clone = torch.maximum(x.clone(),torch.tensor(0.0))
#         xmax = torch.max(x_clone)
#         sigma = (x_clone.pow(2).mean(dim=dim)+1e-5)**0.5
#         r = xmax/torch.min(sigma)
        
#         r = torch.maximum(r,torch.tensor(1.0))
#         loss = torch.log(r)
#         # loss = (torch.max(sigma) - torch.min(sigma)).abs()
#         # loss = r
#         self.append(loss)      


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
            # if module.__class__.__name__ == 'TempReLU':BatchNorm2d
            # if 'batchnorm' in module.__class__.__name__.lower():
            if 'ROE' == module.__class__.__name__:
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
            #self.module_dict[y] = self.module_dict[y].register_full_backward_hook(grad_hook) 
            
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
            