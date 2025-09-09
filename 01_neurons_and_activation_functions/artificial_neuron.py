import numpy as np

class ArtificialNeuron:
    """
    Implements a single artificial neuron.

    The neuron computes: output = activation(sum(weights * inputs) + bias)
    """

    def __init__(self, num_inputs, activation_function='sigmoid'):
        """
        Initialize the neuron with random weights and bias.

        Args:
            num_inputs (int): Number of input features
            activation_function (str): Type of activation ('sigmoid', 'tanh', 'relu')
        """
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.activation_function = activation_function

    def activation(self, z):
        """
        Apply activation function to the weighted sum.

        Args:
            z (float): Weighted sum plus bias

        Returns:
            float: Activated output
        """
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation_function == 'tanh':
            return np.tanh(z)
        elif self.activation_function == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, inputs):
        """
        Compute the forward pass of the neuron.

        Args:
            inputs (np.array): Input features

        Returns:
            float: Neuron output
        """
        z = np.dot(self.weights, inputs) + self.bias
        return self.activation(z)

# Example usage
if __name__ == "__main__":
    # Create a neuron with 3 inputs
    neuron = ArtificialNeuron(num_inputs=3, activation_function='sigmoid')

    # Sample input
    inputs = np.array([0.5, -0.2, 0.8])

    # Compute output
    output = neuron.forward(inputs)
    print(f"Neuron output: {output}")

    # Manual calculation for verification
    z = np.dot(neuron.weights, inputs) + neuron.bias
    manual_output = 1 / (1 + np.exp(-z))
    print(f"Manual calculation: {manual_output}")
    print(f"Match: {np.isclose(output, manual_output)}")
