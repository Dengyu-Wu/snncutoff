# Importing necessary classes from snncutoff.regularizer
from snncutoff.regularizer import RCSANN, RCSSNN, RCSANNLoss, RCSSNNLoss
from typing import Type, Union

class NoneReg:
    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute_reg_loss(self, x, y, features):
        return 0.0 

# Dictionary mapping for ANN regularizers
ann_regularizer: dict[str, Union[Type[NoneReg], RCSANN]] = {
    'none': NoneReg,
    'rcs': RCSANN(),
}

# Dictionary mapping for SNN regularizers
snn_regularizer: dict[str, Union[Type[NoneReg], RCSSNN]] = {
    'none': NoneReg,
    'rcs': RCSSNN(),
}

# Dictionary mapping for ANN regularizer losses
ann_regularizer_loss: dict[str, Type] = {
    'none': NoneReg,
    'rcs': RCSANNLoss,
}

# Dictionary mapping for SNN regularizer losses
snn_regularizer_loss: dict[str, Type] = {
    'none': NoneReg,
    'rcs': RCSSNNLoss,
}

# Function to get the regularizer based on name and method
def get_regularizer(name: str, method: str):
    """
    Retrieve the regularizer based on the provided name and method.

    Args:
        name (str): The name of the regularizer.
        method (str): The method type, either 'ann' or 'snn'.

    Returns:
        Type: The corresponding regularizer class instance.

    Raises:
        KeyError: If the name is not found in the regularizer dictionary.
        ValueError: If the method is not 'ann' or 'snn'.
    """
    if method == 'ann':
        try:
            return ann_regularizer[name]
        except KeyError:
            raise KeyError(f"Invalid regularizer name: {name}. Available names are: {', '.join(ann_regularizer.keys())}")
    elif method == 'snn':
        try:
            return snn_regularizer[name]
        except KeyError:
            raise KeyError(f"Invalid regularizer name: {name}. Available names are: {', '.join(snn_regularizer.keys())}")
    else:
        raise ValueError(f"Invalid method: {method}. Expected 'ann' or 'snn'.")

# Function to get the regularizer loss based on name and method
def get_regularizer_loss(name: str, method: str):
    """
    Retrieve the regularizer loss based on the provided name and method.

    Args:
        name (str): The name of the regularizer loss.
        method (str): The method type, either 'ann' or 'snn'.

    Returns:
        Type: The corresponding regularizer loss class.

    Raises:
        KeyError: If the name is not found in the regularizer loss dictionary.
        ValueError: If the method is not 'ann' or 'snn'.
    """
    if method == 'ann':
        try:
            return ann_regularizer_loss[name]
        except KeyError:
            raise KeyError(f"Invalid regularizer loss name: {name}. Available names are: {', '.join(ann_regularizer_loss.keys())}")
    elif method == 'snn':
        try:
            return snn_regularizer_loss[name]
        except KeyError:
            raise KeyError(f"Invalid regularizer loss name: {name}. Available names are: {', '.join(snn_regularizer_loss.keys())}")
    else:
        raise ValueError(f"Invalid method: {method}. Expected 'ann' or 'snn'.")

# Example usage:
# regularizer = get_regularizer('rcs', 'ann')
# regularizer_loss = get_regularizer_loss('rcs', 'snn')
