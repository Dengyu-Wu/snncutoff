# Importing necessary classes from snncutoff.cutoff
from snncutoff.cutoff import BaseCutoff, ConfCutoff, TopKCutoff, SpikeCPCutoff
from typing import Type

# Dictionary mapping for cutoff classes
cutoff_list: dict[str, Type] = {
    'timestep': BaseCutoff,
    'conf': ConfCutoff,
    'topk': TopKCutoff,
    'spikecp': SpikeCPCutoff,
}

# Function to get the cutoff class based on name
def get_cutoff(name: str) -> Type:
    """
    Retrieve the cutoff class based on the provided name.

    Args:
        name (str): The name of the cutoff class.

    Returns:
        Type: The corresponding cutoff class.

    Raises:
        KeyError: If the name is not found in the cutoff_list.
    """
    try:
        return cutoff_list[name]
    except KeyError:
        raise KeyError(f"Invalid name: {name}. Available names are: {', '.join(cutoff_list.keys())}")

# Example usage:
# cutoff = get_cutoff('timestep')
