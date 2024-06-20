import torch
from torch import nn
import numpy as np

class RCSANN(nn.Module):
    def __init__(self):
        """
        Initialize the RCSANN module.
        """
        super().__init__()
        self.add_loss = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RCSANN module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The computed RCS value.
        """
        x = x.unsqueeze(0)
        rank = len(x.size()) - 2  # N, T, C, W, H
        dim = -np.arange(rank) - 1
        dim = list(dim)
        x_clone = torch.maximum(x, torch.tensor(0.0))
        xmax = torch.max(x_clone)
        sigma = (x_clone.pow(2).mean(dim=dim) + 1e-5) ** 0.5
        r = xmax / sigma
        return r


class RCSANNLoss:
    def __init__(self, *args, **kwargs):
        """
        Initialize the RCSANNLoss module.
        """
        super().__init__()

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
        right_predict_mask = x.max(-1)[1].eq(_target).to(torch.float32)
        right_predict_mask = right_predict_mask.prod(0, keepdim=True)
        right_predict_mask = torch.unsqueeze(right_predict_mask, dim=2).flatten(0, 1).contiguous().detach()
        features = features * right_predict_mask
        features = features.max(dim=0)[0]
        loss = features.mean()
        return loss

# Example usage:
# model = RCSANN()
# loss_fn = RCSANNLoss()
# x = torch.randn(10, 5)  # Example input tensor
# y = torch.randint(0, 5, (10,))  # Example target tensor
# features = torch.randn(10, 5)  # Example features tensor
# output = model(x)
# loss = loss_fn.compute_reg_loss(output, y, features)
# print(loss)
