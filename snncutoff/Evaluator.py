import argparse
import torch
import torch.nn as nn
from typing import Callable, List, Type
from .cutoff import BaseCutoff
from .snncase import SNNCASE

class OutputHook(list):
    def __init__(self):
        self.mask = 0                
    def __call__(self, module, inputs, output):
        loss = []
        loss.append(output[0])
        layer_size = torch.tensor(list(output[0].shape[2:]))
        layer_size = torch.prod(layer_size)
        loss.append(layer_size)
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
            if module.__class__.__name__ == 'LIFSpike':
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
            

class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        data_root: str = './data',
        config_root: str = './configs',
        T: int = 10,
        sigma: float = 1.0,
        postprocessor: Type[BaseCutoff] = None,
        batch_size: int = 200,
        shuffle: bool = False,
        num_workers: int = 4,
        add_time_dim: bool = False
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) discriminative OOD
        detection methods.

        Args:
            net (nn.Module):
                The base classifier.
            id_name (str):
                The name of the in-distribution dataset.
            data_root (str, optional):
                The path of the data folder. Defaults to './data'.
            config_root (str, optional):
                The path of the config folder. Defaults to './configs'.
            preprocessor (Callable, optional):
                The preprocessor of input images.
                Passing None will use the default preprocessor
                following convention. Defaults to None.
            postprocessor_name (str, optional):
                The name of the postprocessor that obtains OOD score.
                Ignored if an actual postprocessor is passed.
                Defaults to None.
            postprocessor (Type[BasePostprocessor], optional):
                An actual postprocessor instance which inherits
                OpenOOD's BasePostprocessor. Defaults to None.
            batch_size (int, optional):
                The batch size of samples. Defaults to 200.
            shuffle (bool, optional):
                Whether shuffling samples. Defaults to False.
            num_workers (int, optional):
                The num_workers argument that will be passed to
                data loaders. Defaults to 4.

        Raises:
            ValueError:
                If both postprocessor_name and postprocessor are None.
            ValueError:
                If the specified ID dataset {id_name} is not supported.
            TypeError:
                If the passed postprocessor does not inherit BasePostprocessor.
        """
        self.net = net
        self.net.eval()
        self.postprocessor=BaseCutoff(T=T,add_time_dim=add_time_dim)
        self.T = T
        self.sigma = sigma


    def evaluation(self,data_loader):
        outputs_list, label_list = self.postprocessor.inference(net=self.net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        outputs_last = torch.softmax(outputs_list[-1],dim=-1)
        outputs_list = torch.softmax(outputs_list,dim=-1)
        acc =(outputs_list.max(-1)[1] == new_label).float().sum(1)/label_list.size()[0]
        new_label  = torch.nn.functional.one_hot(new_label[0], num_classes=10) 
        loss = []
        outputs_list = torch.softmax(outputs_list,dim=-1)
        for t in range(self.T):
            # loss_t = torch.nn.CrossEntropyLoss()(outputs_list[t],new_label.float()) # to ground truth
            loss_t = torch.nn.MSELoss()(outputs_list[t],outputs_last)  # to last timestep
            loss.append(loss_t.cpu().numpy().item())
        return acc.cpu().numpy().tolist(), loss
    
    def aoi_evaluation(self,data_loader):
        outputs_list, label_list = self.postprocessor.inference(net=self.net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        # print(acc)
        outputs_last = torch.softmax(outputs_list[-1],dim=-1)
        index = (outputs_list.max(-1)[1] == new_label).float()
        for t in range(self.T-1,0,-1):
            index[t-1] = index[t]*index[t-1]
        index[-1] = 1.0
        index = torch.argmax(index,dim=0)
        mask = torch.nn.functional.one_hot(index, num_classes=self.T)
        outputs_list = outputs_list*mask.transpose(0,1).unsqueeze(-1)
        outputs_list = outputs_list.sum(0)
        acc = (outputs_list.max(-1)[1]  == new_label[0]).float().sum()/label_list.size()[0]
        outputs_list = torch.softmax(outputs_list,dim=-1)
        new_label  = torch.nn.functional.one_hot(label_list, num_classes=10) 
        # loss = torch.nn.CrossEntropyLoss()(outputs_list,new_label.float()) # to ground truth
        loss = torch.nn.MSELoss()(outputs_list,outputs_last) # to ground truth
        return acc.cpu().numpy().item(), (index+1).cpu().numpy()

    def cutoff_evaluation(self,data_loader):
        outputs_list, label_list = self.postprocessor.inference(net=self.net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        # print(acc)
        outputs_last = torch.softmax(outputs_list[-1],dim=-1)
        index = (outputs_list.max(-1)[1] == new_label).float()
        for t in range(self.T-1,0,-1):
            index[t-1] = index[t]*index[t-1]
        index[-1] = 1.0
        index = torch.argmax(index,dim=0)
        print((index+1).float().mean())
        index = torch.nn.functional.one_hot(index, num_classes=self.T)
        outputs_list = outputs_list*index.transpose(0,1).unsqueeze(-1)
        outputs_list = outputs_list.sum(0)
        acc =(outputs_list.max(-1)[1]  == new_label[0]).float().sum()/label_list.size()[0]
        outputs_list = torch.softmax(outputs_list,dim=-1)
        new_label  = torch.nn.functional.one_hot(label_list, num_classes=10) 
        # loss = torch.nn.CrossEntropyLoss()(outputs_list,new_label.float()) # to ground truth
        loss = torch.nn.MSELoss()(outputs_list,outputs_last) # to ground truth
        return acc.cpu().numpy().item(), loss.cpu().numpy().item()

    def MOPS(self):
            model = self.net
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

            connections = []
            print('MOPS.......')

            output_hook = OutputHook()
            model = sethook(output_hook)(model)
            input_size = (3,32,32)
            inputs = torch.randn(input_size).unsqueeze(0).to(device)
            outputs = model(inputs)
            connections = list(output_hook)
            model = sethook(output_hook)(model,remove=True)

            tot_fp = 0
            tot_bp = 0
            for name,w,output in connections:
                # name = connection[0]
                # w = connection[1]
                # output = connection[2]
                fin = torch.prod(torch.tensor(w))
                N_neuron = torch.prod(torch.tensor(output))
                tot_fp += (fin*2+1)*N_neuron
                tot_bp += 2*fin + (fin*2+1)*N_neuron
            tot_op = self.Nops[0]*tot_fp + self.Nops[1]*tot_bp
            print(tot_op)
            print(tot_fp)
            print(tot_bp)