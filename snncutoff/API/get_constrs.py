from snncutoff.constrs.ann import BaseConstrs, QCFSConstrs, ClipConstrs
from snncutoff.constrs.snn import BaseLayer, SimpleBaseLayer
from typing import Type, Union

# Dictionary mapping for ANN constructors
ann_constrs: dict[str, Type] = {
    'baseconstrs': BaseConstrs,
    'qcfsconstrs': QCFSConstrs,
    'clipconstrs': ClipConstrs,
}

# Dictionary mapping for SNN layers
snn_layers: dict[str, Type] = {
    'baselayer': BaseLayer,
    'simplebaselayer': SimpleBaseLayer,
}

# Function to get the constructor or layer based on name and method
def get_constrs(name: str, method: str) -> Union[Type, None]:
    """
    Retrieve the constructor or layer class based on the provided name and method.
    
    Args:
        name (str): The name of the constructor or layer.
        method (str): The method type, either 'ann' or 'snn'.

    Returns:
        Type: The corresponding constructor or layer class.
        None: If the name or method is invalid.
    """
    if method == 'ann':
        return ann_constrs.get(name)
    elif method == 'snn':
        return snn_layers.get(name)
    else:
        raise ValueError(f"Invalid method: {method}. Expected 'ann' or 'snn'.")