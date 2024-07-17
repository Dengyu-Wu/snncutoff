from torch import nn
from typing import Type
from snncutoff.neuron import LIF

class BaseLayer(nn.Module):
    def __init__(self, 
                 neuron: Type[nn.Module]=LIF,
                 regularizer: Type[nn.Module]=None, 
                 T: int=10, 
                 neuron_params: dict = {'vthr': 1.0, 
                                        'tau': 0.,
                                        'mem_init': 0.,
                                        'multistep': True,
                                        'reset_mode': 'hard',
                                        },
                 **kwargs):

        super(BaseLayer, self).__init__()
        
        self.T = T
        self.vthr = neuron_params['vthr']
        neuron_params['T'] = self.T 
        self.multistep = neuron_params['multistep']
        self.neuron=neuron(**neuron_params)
        self.regularizer = regularizer
        
    def forward(self, x):  
        x = self.reshape(x)
        spike_post, mem_post = self.neuron(x)
        if self.regularizer is not None:
            loss = self.regularizer(spike_post.clone(), mem_post.clone()/self.vthr)
        return spike_post
         
    
    def reshape(self,x):
        if self.multistep:
            batch_size = int(x.shape[0]/self.T)
            new_dim = [self.T, batch_size]
            new_dim.extend(x.shape[1:])
            return x.reshape(new_dim)
        else:
            return x.unsqueeze(0)
        


