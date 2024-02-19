import torch
import torch.nn as nn
from typing import Type
from pydantic import BaseModel
from snncutoff.cutoff import BaseCutoff
from snncutoff.API import get_cutoff
from snncutoff.utils import OutputHook, sethook, set_dropout

class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        args: Type[BaseModel]=None,
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
        

    # def cutoff_evaluation(self,data_loader,train_loader,epsilon=0.0):
    #     net = self.net
    #     net = set_dropout(net,0.3,training=True)
    #     beta, conf = self.cutoff.setup(net=self.net, data_loader=train_loader,epsilon=epsilon)
    #     net = set_dropout(net,training=False)
    #     outputs_list, label_list = self.cutoff.inference(net=self.net, data_loader=data_loader)
    #     new_label = label_list.unsqueeze(0)
    #     topk = torch.topk(outputs_list,2,dim=-1)
    #     topk_gap_t = topk[0][...,0] - topk[0][...,1] 
    #     index = (topk_gap_t>beta.unsqueeze(-1)).float()
    #     index[-1] = 1.0
    #     index = torch.argmax(index,dim=0)
    #     mask = torch.nn.functional.one_hot(index, num_classes=self.T)
    #     outputs_list = outputs_list*mask.transpose(0,1).unsqueeze(-1)
    #     outputs_list = outputs_list.sum(0)
    #     acc = (outputs_list.max(-1)[1]  == new_label[0]).float().sum()/label_list.size()[0]
    #     return acc.cpu().numpy().item(), (index+1).cpu().numpy(), conf

    def ANN_OPS(self,input_size):
            net = self.net
            print('ANN MOPS.......')
            output_hook = OutputHook(get_connection=True)
            net = sethook(output_hook)(net)
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
    
    def SNN_Spike_Count(self,input_size):
            net = self.net
            connections = []
            output_hook = OutputHook(get_connection=True)
            net = sethook(output_hook)(net)
            inputs = torch.randn(input_size).unsqueeze(0).to(net.device)
            outputs = net(inputs)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)

            tot_fp = 0
            tot_bp = 0
            for name,w,output in connections:
                fin = torch.prod(torch.tensor(w))
                N_neuron = torch.prod(torch.tensor(output))
                tot_fp += (fin*2+1)*N_neuron
                tot_bp += 2*fin + (fin*2+1)*N_neuron
            tot_op = self.Nops[0]*tot_fp + self.Nops[1]*tot_bp
            return [tot_op, tot_fp, tot_bp]