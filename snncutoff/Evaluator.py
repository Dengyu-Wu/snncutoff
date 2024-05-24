import torch
import torch.nn as nn
from typing import Type
from pydantic import BaseModel
from snncutoff.cutoff import BaseCutoff
from snncutoff.API import get_cutoff
from snncutoff.utils import OutputHook, sethook, set_dropout
from torch.utils.data import DataLoader
from tqdm import tqdm

class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        args: Type[BaseModel]=None,
        cutoff: Type[BaseCutoff] = None,
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) SNN models.

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
        self.cutoff = cutoff(T=args.T, bin_size=100,add_time_dim=args.add_time_dim, multistep=args.multistep)
        self.T = args.T
        self.add_time_dim = args.add_time_dim

    def evaluation(self,data_loader):
        outputs_list, label_list = self.cutoff.inference(net=self.net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        outputs_list = torch.softmax(outputs_list,dim=-1)
        acc =(outputs_list.max(-1)[1] == new_label).float().sum(1)/label_list.size()[0]
        return acc.cpu().numpy().tolist(), 0.0
    
    def oct_evaluation(self,data_loader):
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

    def cutoff_evaluation(self,data_loader,train_loader,epsilon=0.0):
        acc, timestep, conf = self.cutoff.cutoff_evaluation(net=self.net, 
                                                            data_loader=data_loader,
                                                            train_loader=train_loader,
                                                            epsilon=epsilon)
        return acc, timestep, conf
        

    def ANN_OPS(self,input_size):
            net = self.net
            print('ANN MOPS.......')
            output_hook = OutputHook(output_type='connection')
            net = sethook(output_hook,output_type='connection')(net)
            inputs = torch.randn(input_size).unsqueeze(0).to(net.device)
            outputs = net(inputs)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)
            tot_fp = 0
            for name,w,output in connections:
                fin = torch.prod(torch.tensor(w))
                N_neuron = torch.prod(torch.tensor(output))
                tot_fp += (fin*2+1)*N_neuron
            print(tot_fp)
            return tot_fp
    
    def preprocess(self,x):
        if self.add_time_dim:
            x = x.unsqueeze(1)
            x = x.repeat(1,self.T,1,1,1)
        return x.transpose(0,1)
    
    @torch.no_grad()
    def SNN_SOP_Count(self,
                    data_loader: DataLoader,
                    progress: bool = True):
            net = self.net
            connections = []
            input_size = [10,2,128,128]
            print('get connection......')
            for data, label in tqdm(data_loader,
                            disable=not progress):
                data = data.cuda()
                data = self.preprocess(data)
                output_hook = OutputHook(output_type='connection')
                net = sethook(output_hook,output_type='connection')(net)
                outputs = net(data)
                connections = list(output_hook)
                net = sethook(output_hook)(net,remove=True)
                fin_tot = []
                for name,w,output in connections:
                    fin = torch.prod(torch.tensor(w))
                    fin_tot.append(fin)
                break

            print('SNN SOP.......')
            tot_sop = 0
            i = 0
            for data, label in tqdm(data_loader,
                            disable=not progress):
                data = data.cuda()
                data = self.preprocess(data)
                label = label.cuda()
                output_hook = OutputHook(output_type='activation')
                net = sethook(output_hook,output_type='activation')(net)
                outputs = net(data)
                connections = list(output_hook)
                net = sethook(output_hook)(net,remove=True)
                tot_fp = fin_tot[0]*data.sum()/torch.prod(torch.tensor(data.size()[2:]))

                n = 1
                for output_size, output_spike in connections:
                    fin = fin_tot[n]
                    N_neuron = torch.prod(torch.tensor(input_size))
                    s_avg = output_spike.sum()/N_neuron
                    tot_fp += fin*N_neuron
                    n += 1
                tot_sop += tot_fp
                i += data.size()[1]
            tot_sop = tot_sop/i
            return tot_sop.cpu().numpy().item()
       
    @torch.no_grad()
    def SNN_Spike_Count(self,
                  data_loader: DataLoader,
                  progress: bool = True):
            net = self.net
            connections = []
            print('SNN SOP.......')
            i = 0
            tot_spike = 0
            for data, label in tqdm(data_loader,
                            disable=not progress):
                data = data.cuda()
                data = self.preprocess(data)
                label = label.cuda()
                output_hook = OutputHook(output_type='activation')
                net = sethook(output_hook,output_type='activation')(net)
                outputs = net(data)
                connections = list(output_hook)
                net = sethook(output_hook)(net,remove=True)
                tot_fp = data.sum()

                n = 1
                for output_size, output_spike in connections:
                    tot_fp += output_spike.sum()
                    n += 1
                tot_spike += tot_fp
                i += data.size()[1]
            tot_spike = tot_spike/i
            return tot_spike.cpu().numpy().item()
