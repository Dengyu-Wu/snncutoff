import torch
import torch.nn as nn
from typing import Callable, List, Type
from .cutoff import BaseCutoff
from snncutoff.API import get_cutoff

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
        args=None,
        cutoff: Type[BaseCutoff] = None,
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) discriminative OOD
        detection methods.

        Args:
            net (nn.Module):
                The base classifier.
            cutoff (Type[BaseCutoff], optional):
                An actual cutoff instance which inherits
                SNNCutoff's BaseCutoff. Defaults to None.
        Raises:
            ValueError:
                If both postprocessor_name and postprocessor are None.
            ValueError:
                If the specified ID dataset {id_name} is not supported.
            TypeError:
                If the passed postprocessor does not inherit BasePostprocessor.
        """
        self.net = net
        self.args = args
        self.net.eval()
        cutoff=get_cutoff(args.cutoff_name)
        self.cutoff = cutoff(T=args.T, bin_size=100,add_time_dim=args.add_time_dim,sigma=args.sigma,multistep=args.multistep)
        self.T = args.T
        self.sigma = args.sigma
        self.add_time_dim = args.add_time_dim

    def evaluation(self,data_loader):
        outputs_list, label_list = self.cutoff.inference(net=self.net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        outputs_list = torch.softmax(outputs_list,dim=-1)
        acc =(outputs_list.max(-1)[1] == new_label).float().sum(1)/label_list.size()[0]
        return acc.cpu().numpy().tolist(), 0.0
    
    def aoi_evaluation(self,data_loader):
        outputs_list, label_list = self.cutoff.inference(net=self.net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        index = (outputs_list.max(-1)[1] == new_label).float()
        for t in range(self.T-1,0,-1):
            index[t-1] = index[t]*index[t-1]
        index[-1] = 1.0
        index = torch.argmax(index,dim=0)
        mask = torch.nn.functional.one_hot(index, num_classes=self.T)
        outputs_list = outputs_list*mask.transpose(0,1).unsqueeze(-1)
        outputs_list = outputs_list.sum(0)
        acc = (outputs_list.max(-1)[1]  == new_label[0]).float().sum()/label_list.size()[0]
        return acc.cpu().numpy().item(), (index+1).cpu().numpy()

    def cutoff_evaluation(self,data_loader,train_loader):
        beta, conf = self.cutoff.setup(net=self.net, data_loader=train_loader)
        outputs_list, label_list = self.cutoff.inference(net=self.net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        topk = torch.topk(outputs_list,2,dim=-1)
        topk_gap_t = topk[0][...,0] - topk[0][...,1] 
        index = (topk_gap_t>2*beta.unsqueeze(-1)).float()
        index[-1] = 1.0
        index[:,-1] = 1.0
        index = torch.argmax(index,dim=0)
        mask = torch.nn.functional.one_hot(index, num_classes=self.T)
        outputs_list = outputs_list*mask.transpose(0,1).unsqueeze(-1)
        outputs_list = outputs_list.sum(0)
        acc = (outputs_list.max(-1)[1]  == new_label[0]).float().sum()/label_list.size()[0]
        return acc.cpu().numpy().item(), (index+1).cpu().numpy(), conf

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