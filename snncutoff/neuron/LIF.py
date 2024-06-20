class LIF:
    def __init__(self, vthr: float = 1.0, tau: float = 0.5):
        """
        Initialize the LIF neuron model.

        Args:
            vthr (float): The threshold voltage for spike generation.
            tau (float): The time constant of the membrane potential decay.
        """
        self.t = 0.0
        self.vmem = 0.0
        self.vthr = vthr
        self.tau = tau
        self.gamma = 1.0

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

    def is_spike(self) -> bool:
        """
        Check if the membrane potential has reached the threshold.

        Returns:
            bool: True if the membrane potential has reached or exceeded the threshold, False otherwise.
        """
        return self.vmem >= self.vthr

# Example usage:
# lif_neuron = LIF()
# lif_neuron.initMem(0.5)
# lif_neuron.updateMem(1.0)
# print(lif_neuron.is_spike())
