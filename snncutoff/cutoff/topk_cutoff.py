
from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from snncutoff.utils import reset_neuron
from .base_cutoff import BaseCutoff

class TopKCutoff(BaseCutoff):
    def __init__(self, T, bin_size=100,add_time_dim=False, multistep=False):
        self.T = T
        self.add_time_dim = add_time_dim
        self.bin_size = bin_size 
        self.multistep = multistep

    @torch.no_grad()
    def setup(self, 
              net: nn.Module,
              data_loader: DataLoader,
              epsilon: float = 0.0,
              progress: bool = True):

        conf = []
        outputs, pred, ygaps = [], [], []
        for data, label in tqdm(data_loader,
                          disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            label = label.cuda()
            outputs_b, pred_b, ygaps_b = [], [], []
            self.output = 0.0
            if self.multistep:
                output_t = net(data)
                for t in range(output_t.shape[0]):
                    outputs_b.append(output_t[:t+1].sum(0))
                outputs_b = torch.stack(outputs_b,dim=0)
                pred_b = (outputs_b.softmax(-1).max(-1)[1] == label).float()
                topk = torch.topk(outputs_b,2,dim=-1)
                ygaps_b = topk[0][...,0] - topk[0][...,1]
            else:
                for t in range(self.T):
                    output_t = self.postprocess(net, data[t:t+1])
                    pred_t = (output_t.softmax(-1).max(-1)[1] == label).float()
                    topk = torch.topk(output_t,2,dim=-1)
                    topk_gap_t = topk[0][:,0] - topk[0][:,1] 
                    outputs_b.append(output_t)
                    pred_b.append(pred_t)
                    ygaps_b.append(topk_gap_t)
                net = reset_neuron(net)
            
                outputs_b = torch.stack(outputs_b,dim=0)
                pred_b = torch.stack(pred_b,dim=0)
                ygaps_b = torch.stack(ygaps_b,dim=0)
            outputs.append(outputs_b)
            pred.append(pred_b)
            ygaps.append(ygaps_b)
            
        ##Concatenates all outputs into one tensor
        outputs = torch.cat(outputs,dim=1)
        pred = torch.cat(pred,dim=1)
        ygaps = torch.cat(ygaps,dim=1)
        for t in range(self.T-1,0,-1):
            pred[t-1] = pred[t]*pred[t-1]
                
        ygaps_min = 0
        ygaps_max = ygaps.max()
        ygaps_disrete = (ygaps_max - ygaps_min)/self.bin_size

        beta, samples = [], []
        for m in range(self.bin_size):
            beta_m = m*ygaps_disrete
            sample_m = []
            for t in range(self.T):
                cutoff_sample = (ygaps[t] > beta_m).float()
                sample_m.append(torch.tensor([(cutoff_sample * pred[t]).sum(),cutoff_sample.sum()]))
            sample_m = torch.stack(sample_m,dim=0)
            beta.append(beta_m)
            samples.append(sample_m)
        beta = torch.stack(beta,dim=0)    #m 1 
        samples = torch.stack(samples,dim=0)
        conf = samples[...,0]/samples[...,1]
        conf[-1] = 1.0 
        conf_mask = (conf>=1-epsilon).float()
        beta_index = conf_mask.argmax(0)
        print(beta_index)
        return beta[beta_index], [conf.cpu().numpy(), beta.cpu().numpy(), samples.cpu().numpy()]
    
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

        outputs_list = torch.cat(outputs_list,dim=-2)
        label_list = torch.cat(label_list)
        return outputs_list, label_list

