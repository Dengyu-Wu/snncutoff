from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from snncutoff.utils import reset_neuron, set_dropout
from .base_cutoff import BaseCutoff

class TopKCutoff(BaseCutoff):
    def __init__(self, T: int, bin_size: int = 100, add_time_dim: bool = False, multistep: bool = False):
        """
        Initialize the TopKCutoff class.

        Args:
            T (int): The number of time steps.
            bin_size (int): The bin size for processing.
            add_time_dim (bool): Whether to add a time dimension to the input.
            multistep (bool): Whether to use multistep processing.

        Reference:
            This algorithm is based on the paper "Optimising Event-Driven Spiking Neural Network with Regularisation and Cutoff" available at https://arxiv.org/abs/2301.09522.
        """
        self.T = T
        self.add_time_dim = add_time_dim
        self.bin_size = bin_size
        self.multistep = multistep

    @torch.no_grad()
    def setup(self, 
              net: nn.Module,
              data_loader: DataLoader,
              epsilon: float = 0.0,
              progress: bool = True) -> torch.Tensor:
        """
        Setup method for the TopKCutoff class.
        This method calculates the beta values for the cutoff.

        Args:
            net (nn.Module): The neural network model.
            data_loader (DataLoader): The data loader.
            epsilon (float): The epsilon value for confidence thresholding.
            progress (bool): Whether to show a progress bar.

        Returns:
            torch.Tensor: The beta values and the confidence information.
        """
        conf = []
        outputs, pred, ygaps = [], [], []
        for data, label in tqdm(data_loader, disable=not progress):
            data = data.cuda()
            data = self.preprocess(data)
            label = label.cuda()
            outputs_b, pred_b, ygaps_b = [], [], []
            self.output = 0.0
            if self.multistep:
                output_t = net(data)
                for t in range(output_t.shape[0]):
                    outputs_b.append(output_t[:t + 1].sum(0))
                outputs_b = torch.stack(outputs_b, dim=0)
                pred_b = (outputs_b.softmax(-1).max(-1)[1] == label).float()
                topk = torch.topk(outputs_b, 2, dim=-1)
                ygaps_b = topk[0][..., 0] - topk[0][..., 1]
            else:
                for t in range(self.T):
                    output_t = self.postprocess(net, data[t:t + 1])
                    pred_t = (output_t.softmax(-1).max(-1)[1] == label).float()
                    topk = torch.topk(output_t, 2, dim=-1)
                    topk_gap_t = topk[0][:, 0] - topk[0][:, 1]
                    outputs_b.append(output_t)
                    pred_b.append(pred_t)
                    ygaps_b.append(topk_gap_t)
                net = reset_neuron(net)

                outputs_b = torch.stack(outputs_b, dim=0)
                pred_b = torch.stack(pred_b, dim=0)
                ygaps_b = torch.stack(ygaps_b, dim=0)
            outputs.append(outputs_b)
            pred.append(pred_b)
            ygaps.append(ygaps_b)

        # Concatenates all outputs into one tensor
        outputs = torch.cat(outputs, dim=1)
        pred = torch.cat(pred, dim=1)
        ygaps = torch.cat(ygaps, dim=1)
        for t in range(self.T - 1, 0, -1):
            pred[t - 1] = pred[t] * pred[t - 1]

        ygaps_min = 0
        ygaps_max = ygaps.max()
        ygaps_discrete = (ygaps_max - ygaps_min) / self.bin_size

        beta, samples = [], []
        for m in range(self.bin_size):
            beta_m = m * ygaps_discrete
            sample_m = []
            for t in range(self.T):
                cutoff_sample = (ygaps[t] > beta_m).float()
                sample_m.append(torch.tensor([(cutoff_sample * pred[t]).sum(), cutoff_sample.sum()]))
            sample_m = torch.stack(sample_m, dim=0)
            beta.append(beta_m)
            samples.append(sample_m)
        beta = torch.stack(beta, dim=0)  # m 1 
        samples = torch.stack(samples, dim=0)
        conf = samples[..., 0] / samples[..., 1]
        conf[-1] = 1.0 
        conf_mask = (conf >= 1 - epsilon).float()
        beta_index = conf_mask.argmax(0)
        return beta[beta_index], [conf.cpu().numpy(), beta.cpu().numpy(), samples.cpu().numpy()]

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
                          epsilon: float = 0.0) -> tuple[float, torch.Tensor, list]:
        """
        Evaluate the cutoff model.
        This method uses dropout during training and sets the dropout rate for evaluation.

        Args:
            net (nn.Module): The neural network model.
            data_loader (DataLoader): The data loader for evaluation.
            train_loader (DataLoader): The data loader for training.
            dropout_rate (float): The dropout rate.
            epsilon (float): The epsilon value for confidence thresholding.

        Returns:
            tuple[float, torch.Tensor, list]: The accuracy, index array, and confidence information.
        """
        net = set_dropout(net, p=dropout_rate, training=True)
        beta, conf = self.setup(net=net, data_loader=train_loader, epsilon=epsilon)
        net = set_dropout(net, training=False)
        outputs_list, label_list = self.inference(net=net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        topk = torch.topk(outputs_list, 2, dim=-1)
        topk_gap_t = topk[0][..., 0] - topk[0][..., 1]
        index = (topk_gap_t > beta.unsqueeze(-1)).float()
        index[-1] = 1.0
        index = torch.argmax(index, dim=0)
        mask = torch.nn.functional.one_hot(index, num_classes=self.T)
        outputs_list = outputs_list * mask.transpose(0, 1).unsqueeze(-1)
        outputs_list = outputs_list.sum(0)
        acc = (outputs_list.max(-1)[1] == new_label[0]).float().sum() / label_list.size()[0]
        return acc.cpu().numpy().item(), (index + 1).cpu().numpy(), conf
