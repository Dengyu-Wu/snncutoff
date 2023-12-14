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
from snncutoff.models.VGG_models import *
from snncutoff import data_loaders
from snncutoff.functions import TET_loss
import numpy as np
from configs import BaseConfig, SNNConfig, AllConfig
from omegaconf import DictConfig, OmegaConf
import hydra



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

@torch.no_grad()
def test(models, test_loader, device, T,num_classes):
    import numpy as np
    correct = np.zeros((5,T))
    total = 0
    loss = np.zeros((5,T))
    variance = 0
    for model in models:
        model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs = inputs.transpose(0,1)
        #####Extract hidden spikes ######
        _target_onehot  = torch.nn.functional.one_hot(targets, num_classes=num_classes).to(torch.float32)
        target_onehot  = torch.nn.functional.one_hot(targets, num_classes=num_classes) 
        target_onehot  = target_onehot.to(torch.float32)
        target_onehot = target_onehot.unsqueeze(dim=0).unsqueeze(dim=0)
        outputs = []

        k = 0
        for model in models:
            output = model(inputs)
            _loss = []
            for t in range(output.size()[0]):
                _loss.append(nn.CrossEntropyLoss(reduction='sum')(output[t,...].detach(),_target_onehot.to(device)))
            _loss = torch.stack(_loss,dim=0)
            loss[k,...] += _loss.cpu().numpy()
            k += 1
            outputs.append(output)
        outputs = torch.stack(outputs,dim=1) #  T (Timestep)  M (Model) N (Batch) O (Output)   
        _outputs = nn.Softmax(dim=-1)(outputs)
        #####Create mask ######

        # N M T O  -> N M T -> N T 

        variance += (_outputs-_outputs.mean(1,keepdim=True)).pow(2).sum(-1).mean(1).sum(1)  
        # variance += (outputs-target_onehot).pow(2).sum(-1).mean(1).sum(1)

        # N M T O  -> N M T -> N T 
        # variance += -(outputs*torch.log(outputs+1e-7)/torch.log(torch.tensor(10.0))).sum(-1).mean(1).sum(0)   #entropy method
        total += float(targets.size(0))
        for _k in range(k):
            output = outputs[:,_k]
            for t in range(output.size()[0]):
                mean_out = output[0:t+1,...].mean(0)
                _, predicted = mean_out.cpu().max(1)
                correct[_k,t] += float(predicted.eq(targets.cpu()).sum().item())

        
    final_acc = 100 * correct / total
    variance = variance/total
    loss = loss/total

    print(loss)
    print(final_acc)
    return final_acc,variance


@torch.no_grad()
def cutoff(models, test_loader, device, T,  sigma, spike_gap=False):
    import numpy as np
    correct = np.zeros((5,T))
    #correct = 0
    correct_cutoff = 0
    total = 0
    spike_cnt = np.zeros(T)
    spike_cutoff = 0
    latency = 0
    spike_gap = torch.tensor(spike_gap, dtype=torch.bool)
    for model in models:
        model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        #####Extract hidden spikes ######
        inputs = inputs.transpose(0,1)
        k = 0
        outputs=[]
        for model in models:
            output = model(inputs)
            outputs.append(output)
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
    spike_cnt = spike_cnt / total
    spike_cutoff = spike_cutoff/ total
    return final_acc, spike_cnt, cutoff_acc, spike_cutoff, latency

@hydra.main(version_base=None, config_path='../configs', config_name='default')
def main(cfg: DictConfig):
    all_conf = AllConfig(**cfg['base'],**cfg['SNN'])
    all_conf.nprocs = torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = all_conf.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 #train_dataset, val_dataset = data_loaders.build_dvscifar('cifar-dvs') # change to your path
    train_dataset, val_dataset = data_loaders.get_dvs_loaders(path='/LOCAL/dengyu/dvs_dataset/dvs-cifar10', resize=False)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=all_conf.batch_size,
                                              shuffle=False, num_workers=all_conf.workers, pin_memory=True)
    seed_set = [220, 310, 320, 520, 620]
    models = [VGGSNN() for i in range(len(seed_set))]  
    i= 0
    path = '/users/wooooo/project/SNN-Experiment/'
    dataset='Cifar10DVS/'
    alpha = all_conf.alpha
    for one_seed in seed_set:
        state_dict = torch.load(path+'saved_models/'+dataset+'alpha_'+str(alpha)+'_seed_'+str(one_seed)+'.pth', map_location=torch.device('cpu'))
        models[i].load_state_dict(state_dict, strict=False)
        models[i] = torch.nn.DataParallel(models[i])
        models[i].to(device)
        i += 1
    cutoff_acc_mult = []
    latency_mult = []
    for sigma in np.arange(0.8,1.0,0.01):
        final_acc, spike_cnt, cutoff_acc, spike_cutoff, latency = cutoff(models, test_loader, device, T=10, sigma=sigma)
        cutoff_acc_mult.append(cutoff_acc)
        latency_mult.append(latency)

    # mops_total_mult = [] # [confidence, cutoff by timestep]
    # for sigma in np.arange(0.8,1.0,0.01):
    #     final_acc, mops_total, cutoff_acc, latency = cutoff(models, test_loader, device, T=10, sigma=sigma)
    #     mops_total_mult.append(mops_total)
    # mops_total_mult = np.stack(mops_total_mult,axis=1)
    # dataset='Cifar10DVS/'
    # np.save(path+'Experiment/results/'+dataset+'acc_cutoff_mops_'+str(alpha),mops_total_mult)
    # # np.save(path+'Experiment/results/'+dataset+'acc_cutoff_confidence_'+str(alpha),cutoff_acc_mult)
    # # np.save(path+'Experiment/results/'+dataset+'acc_cutoff_latency_'+str(alpha),latency_mult)

    # final_acc, variance = test(models, test_loader, device, T=10,num_classes=num_classes)
    # dataset='Cifar10DVS/'
    # np.save(path+'Experiment/results/'+dataset+'acc_'+str(alpha),final_acc)
    # np.save(path+'Experiment/results/'+dataset+'var_'+str(alpha),variance.cpu().numpy())

    dataset='Cifar10DVS/'
    np.save(path+'Experiment/results/'+dataset+'acc_cutoff_timestep_'+str(alpha),final_acc)
    np.save(path+'Experiment/results/'+dataset+'acc_cutoff_confidence_'+str(alpha),cutoff_acc_mult)
    np.save(path+'Experiment/results/'+dataset+'acc_cutoff_latency_'+str(alpha),latency_mult)


    print('Test accuracy of the model:', final_acc)
    print('Test cutoff_acc of the model:', cutoff_acc_mult)
    print('Test latency of the model:', latency_mult)

if __name__ == '__main__':
   main()
                                                                                            