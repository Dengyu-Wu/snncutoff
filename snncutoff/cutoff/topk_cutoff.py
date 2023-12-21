
from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from snncutoff.utils import reset_neuron

class TopKCutoff:
    def __init__(self, T, sigma=1.0, bin_size=100,add_time_dim=False):
        # self.config = config
        self.T = T
        self.add_time_dim = add_time_dim
        self.bin_size = bin_size 
        self.sigma = sigma 
        self.beta = None

    @torch.no_grad()
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
            outputs, pred, ygaps = [], [], []
            self.output = 0.0
            for t in range(self.T):
                output_t = self.postprocess(net, data[t:t+1])
                pred_t = (output_t.softmax(-1).max(-1)[1] == label).float()
                topk = torch.topk(output_t,2,dim=-1)
                topk_gap_t = topk[0][:,0] - topk[0][:,1] 
                outputs.append(output_t)
                pred.append(pred_t)
                ygaps.append(topk_gap_t)
                # conf.append(conf_t.cpu())
            net = reset_neuron(net)
            
            outputs = torch.stack(outputs,dim=0)
            pred = torch.stack(pred,dim=0)
            ygaps = torch.stack(ygaps,dim=0)
            for t in range(self.T-1,0,-1):
                pred[t-1] = pred[t]*pred[t-1]
                
            # conf = torch.stack(conf,dim=0)
            ygaps_min = ygaps.min()
            ygaps_max = ygaps.max()
            ygaps_disrete = (ygaps_max - ygaps_min)/self.bin_size
            conf = []
            beta = []
            for m in range(self.bin_size):
                beta_m = m*ygaps_disrete+ygaps_min
                cont_m = []
                for t in range(self.T):
                    cutoff_sample = (ygaps[t] > beta_m).float()
                    conf_t = (cutoff_sample * pred[t]).sum()/cutoff_sample.sum()
                    cont_m.append(conf_t)
                cont_m = torch.stack(cont_m,dim=0)
                conf.append(cont_m)
                beta.append(beta_m)
            conf = torch.stack(conf,dim=0)
            beta = torch.stack(beta,dim=0)
            conf[-1] = 1.0 
            self.beta = beta[(conf>=self.sigma).float().argmax(0)]
        return self.beta
        

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

