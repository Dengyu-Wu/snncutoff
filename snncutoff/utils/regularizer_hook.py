import torch
from snncutoff.neuron import *

class OutputHook(list):
    def __init__(self, get_connection=False):
        self.mask = 0  
        self.get_connection = get_connection
    def __call__(self, module, inputs, output):
        if self.get_connection:
            loss = []
            loss.append(output[0])
            layer_size = torch.tensor(list(output[0].shape[2:]))
            layer_size = torch.prod(layer_size)
            loss.append(layer_size)
            self.append(loss)                  
        else:
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