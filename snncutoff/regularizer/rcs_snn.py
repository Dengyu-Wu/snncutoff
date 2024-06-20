import torch
from torch import nn
import numpy as np


class RCSSNN(nn.Module):
    def __init__(self, beta: float = 0.3):
        """
        Initialize the RCSSNN module.

        Args:
            beta (float): The fraction of the input sequence to consider for the delayed response.
        """
        super().__init__()
        self.beta = beta
        self.add_loss = True

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RCSSNN module.

        Args:
            x (torch.Tensor): The input tensor.
            mem (torch.Tensor): The membrane potential tensor.

        Returns:
            torch.Tensor: The computed cosine similarity inverse.
        """
        rank = len(x.size()) - 2  # T, N, C, W, H
        dim = -np.arange(rank) - 1
        dim = list(dim)
        r_t = x + torch.relu(mem)
        index = -int(r_t.shape[0] * self.beta)
        r_d = r_t[index:, ...].mean(0, keepdim=True).detach()
        r_d_norm = (r_d.pow(2).sum(dim=dim) + 1e-5) ** 0.5
        r_t_norm = (r_t.pow(2).sum(dim=dim) + 1e-5) ** 0.5
        cs = (r_t * r_d).sum(dim=dim) / (r_t_norm * r_d_norm)
        cs = cs.mean(0, keepdim=True)
        cs = 1 / (cs + 1e-5)
        return cs


class RCSSNNLoss:
    def __init__(self, config, *args, **kwargs):
        """
        Initialize the RCSSNNLoss module.

        Args:
            config: Configuration object containing the rcs_n attribute.
        """
        super().__init__()
        self.rcs_n = config.rcs_n

    def compute_reg_loss(self, x: torch.Tensor, y: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the regularization loss.

        Args:
            x (torch.Tensor): The predictions.
            y (torch.Tensor): The target labels.
            features (torch.Tensor): The feature activations.

        Returns:
            torch.Tensor: The computed regularization loss.
        """
        _target = torch.unsqueeze(y, dim=0)
        index = -int(x.shape[0] * self.rcs_n)
        right_predict_mask = x[index:].max(-1)[1].eq(_target).to(torch.float32)
        right_predict_mask = right_predict_mask.prod(0, keepdim=True)
        right_predict_mask = torch.unsqueeze(right_predict_mask, dim=2).flatten(0, 1).contiguous().detach()
        features = features * right_predict_mask
        features = features.max(dim=0)[0]
        loss = features.mean()
        return loss

# Example usage:
# config = type('config', (object,), {'rcs_n': 0.3})()  # Example configuration object
# model = RCSSNN()
# loss_fn = RCSSNNLoss(config)
# x = torch.randn(10, 5)  # Example input tensor
# mem = torch.randn(10, 5)  # Example membrane potential tensor
# y = torch.randint(0, 5, (10,))  # Example target tensor
# features = torch.randn(10, 5)  # Example features tensor
# output = model(x, mem)
# loss = loss_fn.compute_reg_loss(output, y, features)
# print(loss)
