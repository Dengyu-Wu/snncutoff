from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from snncutoff.utils import reset_neuron

class BaseCutoff:
    def __init__(self, T: int, add_time_dim: bool = False, multistep: bool = False):
        """
        Initialize the BaseCutoff class.

        Args:
            T (int): The number of time steps.
            add_time_dim (bool): Whether to add a time dimension to the input.
            multistep (bool): Whether to use multistep processing.
        """
        self.T = T
        self.add_time_dim = add_time_dim
        self.multistep = multistep

    def setup(self, net: nn.Module, data_loader: DataLoader):
        """
        Setup method for the BaseCutoff class.

        Args:
            net (nn.Module): The neural network model.
            data_loader (DataLoader): The data loader.
        """
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any) -> torch.Tensor:
        """
        Post-process the network output.

        Args:
            net (nn.Module): The neural network model.
            data (Any): The input data.

        Returns:
            torch.Tensor: The processed output.
        """
        output = net(data)
        self.output = self.output + output[0]
        return self.output

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-process the input data.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The preprocessed tensor.
        """
        if self.add_time_dim:
            x = x.unsqueeze(1)
            x = x.repeat(1, self.T, 1, 1, 1)
        return x.transpose(0, 1)

    @torch.no_grad()
    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform inference on the data loader.

        Args:
            net (nn.Module): The neural network model.
            data_loader (DataLoader): The data loader.
            progress (bool): Whether to show a progress bar.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The outputs and labels.
        """
        outputs_list, label_list = [], []
        for data, label in tqdm(data_loader, disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            label = label.cuda()
            outputs = []
            self.output = 0.0

            if self.multistep:
                output_t = net(data)
                for t in range(output_t.shape[0]):
                    outputs.append(output_t[:t + 1].sum(0))
            else:
                for t in range(self.T):
                    output_t = self.postprocess(net, data[t:t + 1])
                    outputs.append(output_t)
                net = reset_neuron(net)
            outputs = torch.stack(outputs, dim=0)
            outputs_list.append(outputs)
            label_list.append(label)

        outputs_list = torch.cat(outputs_list, dim=1)
        label_list = torch.cat(label_list)

        return outputs_list, label_list

# Example usage:
# net = YourNeuralNetwork()
# cutoff = BaseCutoff(T=10, add_time_dim=True, multistep=True)
# data_loader = DataLoader(your_dataset)
# outputs, labels = cutoff.inference(net, data_loader)