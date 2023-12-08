
from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from easycutoff.utils import reset_neuron

class BaseCutoff:
    def __init__(self, T):
        # self.config = config
        self.T = T

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        self.output = self.output + output[0]
        # conf, pred = torch.max(score, dim=-1)
        # self.score = score
        return self.output
    
    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        # pred_list, conf_list, label_list = [], [], []
        outputs_list, label_list = [], []
        for data, label in tqdm(data_loader,
                          disable=not progress):
            data = data.cuda()
            data = data.transpose(0,1).repeat(10,1,1,1,1)
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