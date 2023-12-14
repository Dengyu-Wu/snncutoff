
from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from snncutoff.utils import reset_neuron

class GapCutoff:
    def __init__(self, T, add_time_dim=False):
        # self.config = config
        self.T = T
        self.add_time_dim = add_time_dim

    def setup(self, 
              net: nn.Module,
              data_loader: DataLoader,
              progress: bool = True):
         
        for data, label in tqdm(data_loader,
                          disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            label = label.cuda()
            # pred, conf = [], []
            outputs = []
            self.output = 0.0
            for t in range(self.T):
                output_t = self.postprocess(net, data[t:t+1])
                outputs.append(output_t)
                # conf.append(conf_t.cpu())
            net = reset_neuron(net)
            
            outputs = torch.stack(outputs,dim=0)
            # conf = torch.stack(conf,dim=0)
            
        return 0.0       
        

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        self.output = self.output + output[0]
        # conf, pred = torch.max(score, dim=-1)
        # self.score = score
        return self.output
    
    def preprocess(self,x):
        if self.add_time_dim:
            x = x.unsqueeze(1)
            x = x.repeat(1,self.T,1,1,1)
        return x.transpose(0,1)
    
    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        # pred_list, conf_list, label_list = [], [], []
        outputs_list, label_list = [], []
        for data, label in tqdm(data_loader,
                          disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            label = label.cuda()
            # pred, conf = [], []
            outputs = []
            self.output = 0.0
            for t in range(self.T):
                output_t = self.postprocess(net, data[t:t+1])
                outputs.append(output_t)
                # conf.append(conf_t.cpu())
            net = reset_neuron(net)
            
            outputs = torch.stack(outputs,dim=0)
            # conf = torch.stack(conf,dim=0)
            outputs_list.append(outputs)
            # pred_list.append(outputs.cpu())
            # conf_list.append(conf.cpu())
            label_list.append(label)

        # convert values into numpy array
        outputs_list = torch.cat(outputs_list,dim=-2)#.cpu().numpy()
        # conf_list = torch.cat(conf_list,dim=-1).numpy()
        label_list = torch.cat(label_list)#.cpu().numpy()

        return outputs_list, label_list

