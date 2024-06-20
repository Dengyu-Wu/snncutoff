# Importing necessary classes from snncutoff.loss
from snncutoff.loss import MeanLoss, TETLoss
from typing import Type

# Dictionary mapping for loss functions
loss: dict[str, Type] = {
    'none': None,
    'mean': MeanLoss,
    'tet': TETLoss,
}

# Function to get the loss function based on name and method
def get_loss(name: str, method: str):
    """
    Retrieve the loss function based on the provided name and method.

    Args:
        name (str): The name of the loss function.
        method (str): The method type, either 'ann' or 'snn'.

    Returns:
        Type: The corresponding loss function class or None.

    Raises:
        KeyError: If the name is not found in the loss dictionary.
        ValueError: If the method is not 'ann' or 'snn'.
    """
    if method == 'ann':
        return loss['mean']
    elif method == 'snn':
        try:
            return loss[name]
        except KeyError:
            raise KeyError(f"Invalid loss name: {name}. Available names are: {', '.join(loss.keys())}")
    else:
        raise ValueError(f"Invalid method: {method}. Expected 'ann' or 'snn'.")

# Example usage:
# loss_function = get_loss('mean', 'ann')
# loss_function = get_loss('tet', 'snn')
