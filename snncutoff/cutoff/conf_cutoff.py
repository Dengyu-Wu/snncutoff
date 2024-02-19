
from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from snncutoff.utils import reset_neuron
from .base_cutoff import BaseCutoff

class ConfCutoff(BaseCutoff):
    def __init__(self, T, add_time_dim=False, multistep=False,  *args, **kwargs):
        self.T = T
        self.add_time_dim = add_time_dim
        self.multistep = multistep
        
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        self.output = self.output + output[0]
        return self.output
    
    def preprocess(self,x):
        if self.add_time_dim:
            x = x.unsqueeze(1)
            x = x.repeat(1,self.T,1,1,1)
        return x.transpose(0,1)
    
    @torch.no_grad()
    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):

        outputs_list, label_list = [], []
        for data, label in tqdm(data_loader,
                          disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            label = label.cuda()
            outputs = []
            self.output = 0.0

            if self.multistep:
                output_t = net(data)
                for t in range(output_t.shape[0]):
                    outputs.append(output_t[:t+1].sum(0))
            else:
                for t in range(self.T):
                    output_t = self.postprocess(net, data[t:t+1])
                    outputs.append(output_t)
                net = reset_neuron(net)
            outputs = torch.stack(outputs,dim=0)
            outputs_list.append(outputs)
            label_list.append(label)

        outputs_list = torch.cat(outputs_list,dim=1)
        label_list = torch.cat(label_list)

        return outputs_list, label_list
    
    @torch.no_grad()
    def cutoff_evaluation(self,
                          net: nn.Module,
                          data_loader,
                          train_loader,
                          dropout_rate=0.3,
                          epsilon=0.0):
        
        outputs_list, label_list = self.inference(net=net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        conf = outputs_list.softmax(-1)
        index = (conf.max(-1)[0]>(1.0-epsilon)).float()
        index[-1] = 1
        index = torch.argmax(index,dim=0)
        mask = torch.nn.functional.one_hot(index, num_classes=self.T)
        outputs_list = outputs_list*mask.transpose(0,1).unsqueeze(-1)
        outputs_list = outputs_list.sum(0)
        acc = (outputs_list.max(-1)[1]  == new_label[0]).float().sum()/label_list.size()[0]
        return acc.cpu().numpy().item(), (index+1).cpu().numpy(), conf
