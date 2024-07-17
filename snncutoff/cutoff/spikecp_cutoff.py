from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from snncutoff.utils import reset_neuron, set_dropout
from .base_cutoff import BaseCutoff

class SpikeCPCutoff(BaseCutoff):
    def __init__(self, T: int, bin_size: int = 100, add_time_dim: bool = False, multistep: bool = False):
        """
        Initialize the SpikeCPCutoff class.

        Args:
            T (int): The number of time steps.
            bin_size (int): The bin size for processing.
            add_time_dim (bool): Whether to add a time dimension to the input.
            multistep (bool): Whether to use multistep processing.

        Reference:
            This algorithm is based on the paper "SpikeCP: Delay-Adaptive Reliable Spiking Neural Networks via Conformal Prediction" available at https://arxiv.org/abs/2305.11322.
        """
        self.T = T
        self.add_time_dim = add_time_dim
        self.bin_size = bin_size
        self.multistep = multistep

    @torch.no_grad()
    def setup(self, 
              net: nn.Module,
              data_loader: DataLoader,
              alpha: float = 0.0,
              progress: bool = True) -> torch.Tensor:
        """
        Setup method for the SpikeCPCutoff class.
        This method calculates the score threshold (s_th) for the cutoff.

        Args:
            net (nn.Module): The neural network model.
            data_loader (DataLoader): The data loader.
            alpha (float): The alpha value for score thresholding.
            progress (bool): Whether to show a progress bar.

        Returns:
            torch.Tensor: The score threshold (s_th).
        """
        conf = []
        outputs, nc_score = [], []
        for data, label in tqdm(data_loader, disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            label = label.cuda()
            outputs_b, nc_score_b = [], []
            self.output = 0.0
            if self.multistep:
                output_t = net(data)
                for t in range(output_t.shape[0]):
                    outputs_b.append(output_t[:t + 1].sum(0))
                outputs_b = torch.stack(outputs_b, dim=0)
                nc_score_b = -torch.log(outputs_b.softmax(-1)[:, label] + 1e-5)
            else:
                for t in range(self.T):
                    output_t = self.postprocess(net, data[t:t + 1])
                    nc_score_t = -torch.log(torch.gather(output_t.softmax(-1), dim=-1, index=label.unsqueeze(-1)) + 1e-5)
                    outputs_b.append(output_t)
                    nc_score_b.append(nc_score_t)
                net = reset_neuron(net)

                outputs_b = torch.stack(outputs_b, dim=0)
                nc_score_b = torch.stack(nc_score_b, dim=0)
            outputs.append(outputs_b)
            nc_score.append(nc_score_b)

        # Concatenate all outputs into one tensor
        outputs = torch.cat(outputs, dim=1)
        nc_score = torch.cat(nc_score, dim=1)
        nc_score = torch.cat((nc_score, (nc_score[:, 0] + torch.inf).unsqueeze(1)), dim=1)
        smallest_i = torch.ceil(torch.tensor((1 - alpha) * (nc_score.size()[1] + 1)).to(nc_score.device))
        s_th = -1 * torch.topk(-1 * nc_score, smallest_i.to(torch.int32)-1, dim=1)[0][:, -1, 0]
        return s_th

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

        outputs_list = torch.cat(outputs_list, dim=-2)
        label_list = torch.cat(label_list)
        return outputs_list, label_list

    @torch.no_grad()
    def cutoff_evaluation(self,
                          net: nn.Module,
                          data_loader: DataLoader,
                          train_loader: DataLoader,
                          dropout_rate: float = 0.3,
                          i_th: int = 1,
                          epsilon: float = 0.0) -> tuple[float, torch.Tensor, float]:
        """
        Evaluate the cutoff model.
        This method uses dropout during training and sets the dropout rate for evaluation.

        Args:
            net (nn.Module): The neural network model.
            data_loader (DataLoader): The data loader for evaluation.
            train_loader (DataLoader): The data loader for training.
            dropout_rate (float): The dropout rate.
            i_th (int): The threshold index.
            epsilon (float): The epsilon value for confidence thresholding.

        Returns:
            tuple[float, torch.Tensor, float]: The accuracy, index array, and confidence tensor.

        Reference:
            This algorithm is based on the paper "ConfCutoff: Robust and Efficient Confident Predictions in Deep Neural Networks" available at https://arxiv.org/abs/2305.11322.
        """
        net = set_dropout(net, p=dropout_rate, training=True)
        s_th = self.setup(net=net, data_loader=train_loader, alpha=epsilon)
        net = set_dropout(net, training=False)
        outputs_list, label_list = self.inference(net=net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        nc_score = -(outputs_list.softmax(-1) + 1e-5).log()
        pred_set = (nc_score < s_th.unsqueeze(-1).unsqueeze(-1)).to(torch.float32)
        set_size = pred_set.sum(-1)
        index = set_size.le(i_th).to(torch.float32)
        index[-1] = 1.0
        index = torch.argmax(index, dim=0)
        mask = torch.nn.functional.one_hot(index, num_classes=self.T)
        outputs_list = outputs_list * mask.transpose(0, 1).unsqueeze(-1)
        outputs_list = outputs_list.sum(0)
        acc = (outputs_list.max(-1)[1] == new_label[0]).float().sum() / label_list.size()[0]
        return acc.cpu().numpy().item(), (index + 1).cpu().numpy(), 0.0
