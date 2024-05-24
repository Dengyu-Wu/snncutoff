import torch
from snncutoff.neuron import *

class OutputHook(list):
    def __init__(self, output_type='connection'):
        self.mask = 0  
        self.output_type = output_type
    def __call__(self, module, inputs, output):
        if self.output_type=='connection':
            self.append([module.__class__.__name__,module.weight.size(), output.size()])
        elif self.output_type=='activation':
            output = output.sum([0,1])
            self.append([output.size(), output])
        elif self.output_type=='reg_loss':
            loss = output
            self.append(loss)        

class sethook(object):
    def __init__(self,output_hook,output_type='reg_loss'):
        self.module_dict = {}
        self.k = 0
        self.output_hook = output_hook
        type = {
        'activation': 'neuron',
        'connection': 'connection',
        'reg_loss': 'add_loss',
        }
        self.output_type = type[output_type]
    def get_module(self,model):
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                model._modules[name] = self.get_module(module)
            if self.output_type == 'connection':
                if 'conv' in module.__class__.__name__.lower() or 'linear' in module.__class__.__name__.lower():
                    if 'layer' not  in module.__class__.__name__.lower():
                        self.module_dict[str(self.k)] = module
                        self.k+=1
            else:
                if hasattr(module, self.output_type):
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