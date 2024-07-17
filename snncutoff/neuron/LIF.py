import torch
import torch.nn as nn
from torch.autograd import Function
from snncutoff.gradients import ZIF
from typing import Type


class LIF(nn.Module):
    def __init__(self, 
                 T: int = 4, 
                 vthr: float = 1.0, 
                 tau: float = 0.5, 
                 mem_init: float = 0., 
                 surogate: Type[Function] = ZIF,
                 multistep: bool=True,
                 reset_mode: str = 'hard',
                 *args, 
                 **kwargs):

        super(LIF, self).__init__()
        
        """
        Initialize the LIF neuron model.

        Args:
            vthr (float): The threshold voltage for spike generation.
            tau (float): The time constant of the membrane potential decay.
        """
        self.t = 0.0
        self.T = T
        self.mem_init=mem_init
        self.vmem = 0.0
        self.vthr = vthr
        self.tau = tau
        self.gamma = 1.0
        self.reset_mode = reset_mode
        self.multistep=multistep
        self.surogate = surogate.apply

    def _mem_update_multistep(self, x):  
        spike_post = []
        mem_post = []
        self.reset()
        for t in range(self.T):
            vmem = self.vmem + x[t]
            spike =  self.surogate(vmem - self.vthr, self.gamma)
            vmem = self.vmem_reset(vmem,spike)
            self.updateMem(vmem)
            spike_post.append(spike)
            mem_post.append(vmem)
        return torch.stack(spike_post,dim=0), torch.stack(mem_post,dim=0)

    def _mem_update_singlestep(self,x):
        if self.neuron.t == 0:
            self.mem_init = 0.5 if self.reset_mode == 'soft' else self.mem_init
            self.neuron.initMem(self.mem_init*self.vthr)
        spike_post = []
        vmem = self.neuron.vmem + x[0]
        spike =  (vmem > self.vthr).float()
        vmem = self.vmem_reset(vmem,spike)
        self.neuron.updateMem(vmem)
        spike_post.append(spike*self.vthr)
        return torch.stack(spike_post,dim=0), 0.0

    def forward(self,x):
        if self.multistep:
            return self._mem_update_multistep(x)
        else:
            return self._mem_update_singlestep(x)   


    def reset(self):
        """
        Reset the membrane potential and time step to initial values.
        """
        self.t = 0.0
        self.vmem = 0.0

    def initMem(self, x: float):
        """
        Initialize the membrane potential with a given value.

        Args:
            x (float): The initial membrane potential.
        """
        self.vmem = x

    def updateMem(self, x: float):
        """
        Update the membrane potential based on the input and time constant.

        Args:
            x (float): The input value to update the membrane potential.
        """
        self.vmem = x * self.tau
        self.t += 1

    def is_spike(self) -> bool:
        """
        Check if the membrane potential has reached the threshold.

        Returns:
            bool: True if the membrane potential has reached or exceeded the threshold, False otherwise.
        """
        return self.vmem >= self.vthr

    def vmem_reset(self, x, spike):
        if self.reset_mode == 'hard':
            return x * (1-spike)
        elif self.reset_mode == 'soft':  
            return x - self.vthr*spike
