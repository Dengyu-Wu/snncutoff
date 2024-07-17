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
                 mem_init: float = 0.0, 
                 surrogate: Type[Function] = ZIF,
                 multistep: bool = True,
                 reset_mode: str = 'hard',
                 **kwargs):
        """
        Initialize the LIF neuron model.

        Args:
            T (int): The number of time steps.
            vthr (float): The threshold voltage for spike generation.
            tau (float): The time constant of the membrane potential decay.
            mem_init (float): The initial membrane potential.
            surrogate (Type[Function]): The surrogate gradient function.
            multistep (bool): Whether to use multistep processing.
            reset_mode (str): The mode of resetting the membrane potential ('hard' or 'soft').
        """
        super(LIF, self).__init__()
        self.T = T
        self.vthr = vthr
        self.tau = tau
        self.mem_init = mem_init
        self.vmem = 0.0
        self.gamma = 1.0
        self.reset_mode = reset_mode
        self.multistep = multistep
        self.surrogate = surrogate.apply
        self.t = 0.0

    def _mem_update_multistep(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the membrane potential and generate spikes for multistep input.

        Args:
            x (torch.Tensor): The input tensor over multiple time steps.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The spikes and membrane potentials over time.
        """
        spike_post = []
        mem_post = []
        self.reset()
        for t in range(self.T):
            vmem = self.vmem + x[t]
            spike = self.surrogate(vmem - self.vthr, self.gamma)
            vmem = self.vmem_reset(vmem, spike)
            self.updateMem(vmem)
            spike_post.append(spike)
            mem_post.append(vmem)
        return torch.stack(spike_post, dim=0), torch.stack(mem_post, dim=0)

    def _mem_update_singlestep(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the membrane potential and generate spikes for single-step input.

        Args:
            x (torch.Tensor): The input tensor for a single time step.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The spikes and membrane potentials.
        """
        if self.t == 0:
            self.mem_init = 0.5 if self.reset_mode == 'soft' else self.mem_init
            self.initMem(self.mem_init * self.vthr)
        spike_post = []
        vmem = self.vmem + x[0]
        spike = (vmem > self.vthr).float()
        vmem = self.vmem_reset(vmem, spike)
        self.updateMem(vmem)
        spike_post.append(spike * self.vthr)
        return torch.stack(spike_post, dim=0), 0.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the LIF neuron model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The spikes and membrane potentials.
        """
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

    def vmem_reset(self, x: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        """
        Reset the membrane potential based on the reset mode.

        Args:
            x (torch.Tensor): The membrane potential tensor.
            spike (torch.Tensor): The spike tensor.

        Returns:
            torch.Tensor: The reset membrane potential.
        """
        if self.reset_mode == 'hard':
            return x * (1 - spike)
        elif self.reset_mode == 'soft':
            return x - self.vthr * spike
