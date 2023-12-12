import argparse
import shutil
import os
import time
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from snncutoff.models.resnet_models import resnet19
from snncutoff.models.vggsnns import *
from snncutoff import data_loaders
from snncutoff.functions import TET_loss
import numpy as np
from configs import *
from omegaconf import DictConfig, OmegaConf
import hydra
from snncutoff.utils import replace_activation_by_neuron, ann_to_snn_conversion, reset_neuron
from typing import Callable, List, Type
from .cutoff import BaseCutoff

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
        preprocessor: Callable = None,
        postprocessor_name: str = None,
        postprocessor: Type[BaseCutoff] = None,
        batch_size: int = 200,
        shuffle: bool = False,
        num_workers: int = 4,
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
        self.postprocessor=BaseCutoff(T=10)
        self.T = 10

    def evaluation(self,data_loader):
        outputs_list, label_list = self.postprocessor.inference(net=self.net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        outputs_last = torch.softmax(outputs_list[-1],dim=-1)

        acc =(outputs_list.max(-1)[1] == new_label).float().sum(1)/label_list.size()[0]
        new_label  = torch.nn.functional.one_hot(new_label[0], num_classes=10) 
        loss = []
        outputs_list = torch.softmax(outputs_list,dim=-1)
        for t in range(10):
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


@torch.no_grad()
def mops(models, test_loader, device, T,  sigma, spike_gap=False):
    import numpy as np
    correct = np.zeros((5,T))
    #correct = 0
    correct_cutoff = 0
    total = 0
    # input_mops_total = 0
    mops_total = 0
    latency = 0
    spike_gap = torch.tensor(spike_gap, dtype=torch.bool)
    for model in models:
        model.eval()

    connections = []
    for name, param in model.named_parameters():
        if 'weight' in name and '1.weight' not in name:
            connections.append(torch.tensor(param.size()).prod())
    connections = torch.stack(connections,dim=0)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        #####Extract hidden spikes ######
        inputs = inputs.transpose(0,1)
        k = 0
        outputs=[]
        spikes_models = []
        for model in models:
            output_hook = OutputHook()
            model = sethook(output_hook)(model)
            output = model(inputs)
            outputs.append(output)
            spikes_models.append(output_hook)
            model = sethook(output_hook)(model,remove=True)
            k+=1
        outputs = torch.stack(outputs,dim=1) #  T (Timestep)  M (Model) N (Batch) O (Output)   
        _outputs = nn.Softmax(dim=-1)(outputs)
        sigma_max = torch.ge(_outputs.max(-1)[0], sigma).to(torch.float32) 
        sigma_max[-1,...] = 1
        index = torch.argmax(sigma_max,dim=0) 
        latency += torch.sum(index+1,dim=1)
        mask =[]
        for _ in range(T):
            step = index >= 0
            step = step.to(torch.float32) 
            mask.append(step)
            index = index - 1
        _latency = torch.stack(mask,dim=0)

        input_mult = inputs.unsqueeze(1).repeat(1,len(models),1,1,1,1)
        input_mask = _latency
        input_connections = connections[0]/torch.prod(torch.tensor(inputs.size()[2:]))
        for _ in range(len(input_mult.size()) - len(input_mask.size())):
            input_mask = torch.unsqueeze(input_mask,dim=-1)    
        input_conf_mops = (input_mask*input_mult).mean(1).sum()*input_connections
        input_total_mops = [input_conf_mops]
        for t in range(inputs.size()[0]):
            input_total_mops.append(inputs[0:t+1].sum()*input_connections)
        input_total_mops = torch.stack(input_total_mops,dim=0)
        # input_mops_total += _input_total_mops

        spikes_sum_models = []
        k = 0
        for spikes_model in spikes_models: 
            spikes_layer = []
            output_shape = []
            for m in range(len(spikes_model)):
                spike_layer_m = spikes_model[m][0]
                output_shape_layer_m = spikes_model[m][1]
                _mask = _latency[:,k]

                for _ in range(len(spike_layer_m.size()) - len(_mask.size())):
                    _mask = torch.unsqueeze(_mask,dim=-1)

                spike_layer_m_cutoff = spike_layer_m*_mask
                spike_layer_m_cutoff = spike_layer_m_cutoff.sum()
                spike_layer_m_sum = [spike_layer_m_cutoff]

                for t in range(spike_layer_m.size()[0]):
                    spike_layer_m_sum.append(spike_layer_m[0:t+1].sum())
                spike_layer_m_sum = torch.stack(spike_layer_m_sum,dim=0)

                spikes_layer.append(spike_layer_m_sum)
                output_shape.append(output_shape_layer_m)
            spikes_layer = torch.stack(spikes_layer,dim=1)
            spikes_sum_models.append(spikes_layer)
            k += 1
        spikes_sum_models = torch.stack(spikes_sum_models,dim=1)
        output_shape = torch.stack(output_shape,dim=0)
        spikes_mean_models = spikes_sum_models.mean(1)
        hidden_connections = connections[1:]/output_shape
        hidden_mops_total = (spikes_mean_models*hidden_connections.to(device)).sum(1)
        # mac = (2*model_connections[:-1]+1)*model_connections[1:]
        mops_total += hidden_mops_total+input_total_mops
        # print(hidden_mops_total)
        _latency = torch.unsqueeze(_latency,dim=-1)

        mean_out_cutoff = outputs  * _latency # sigma_max.unsqueeze(-1)
        mean_out_cutoff = mean_out_cutoff.sum(0)
        #mean_out_cutoff = vmem[:,index,1:].sum(1)
        ########################
        #model = sethook(output_hook)(model,remove=True)

        total += float(targets.size(0))
        for _k in range(k):
            output = outputs[:,_k]
            for t in range(output.size()[0]):
                mean_out = output[0:t+1,...].mean(0)
                _, predicted = mean_out.cpu().max(-1)
                correct[_k,t] += float(predicted.eq(targets.cpu()).sum().item())

        _, predicted_cutoff = mean_out_cutoff.cpu().max(-1)
        correct_cutoff += predicted_cutoff.eq(targets.unsqueeze(0)).to(torch.float32).sum(1)     
        
        # _, predicted = mean_out.cpu().max(1)
        # correct += float(predicted.eq(targets).sum().item())
        # if batch_idx % 100 == 0:
        #     acc = 100. * float(correct[-1]) / float(total)
        #     print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)

    final_acc = 100 * correct / total
    cutoff_acc = 100 * correct_cutoff.cpu().numpy() / total
    latency = latency.cpu().numpy()/total
    mops_total = mops_total.cpu().numpy()/total
    # hidden_mops_total = input_mops_total.cpu().numpy() / total
    return final_acc, mops_total, cutoff_acc, latency
