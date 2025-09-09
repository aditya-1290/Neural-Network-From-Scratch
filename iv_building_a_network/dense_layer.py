import numpy as np

class DenseLayer:
    """
    Fully-connected (Dense) layer implementation.
    """

    def __init__(self, input_size, output_size, activation='relu'):
        """
        Initialize the dense layer.

        Args:
            input_size (int): Number of input features
            output_size (int): Number of output features
            activation (str): Activation function ('relu', 'sigmoid', 'tanh', 'linear')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

        # Gradients
        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

        # Cache for backpropagation
        self.input_cache = None
        self.output_cache = None

    def forward(self, input_data):
        """
        Forward pass through the layer.

        Args:
            input_data (np.array): Input data of shape (batch_size, input_size)

        Returns:
            np.array: Output data of shape (batch_size, output_size)
        """
        self.input_cache = input_data
        z = np.dot(input_data, self.weights) + self.bias
        self.output_cache = self._activate(z)
        return self.output_cache

    def backward(self, d_output):
        """
        Backward pass through the layer.

        Args:
            d_output (np.array): Gradient of loss with respect to output

        Returns:
            np.array: Gradient of loss with respect to input
        """
        batch_size = d_output.shape[0]

        # Gradient with respect to activation output
        d_activation = d_output * self._activate_derivative(self.output_cache)

        # Gradient with respect to weights and bias
        self.d_weights = np.dot(self.input_cache.T, d_activation) / batch_size
        self.d_bias = np.sum(d_activation, axis=0, keepdims=True) / batch_size

        # Gradient with respect to input
        d_input = np.dot(d_activation, self.weights.T)

        return d_input

    def update(self, learning_rate):
        """
        Update weights and biases using gradients.

        Args:
            learning_rate (float): Learning rate
        """
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias

    def _activate(self, z):
        """
        Apply activation function.
        """
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _activate_derivative(self, output):
        """
        Compute derivative of activation function.
        """
        if self.activation == 'relu':
            return (output > 0).astype(float)
        elif self.activation == 'sigmoid':
            return output * (1 - output)
        elif self.activation == 'tanh':
            return 1 - output**2
        elif self.activation == 'linear':
            return np.ones_like(output)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

# Example usage
if __name__ == "__main__":
    # Create a dense layer
    layer = DenseLayer(input_size=4, output_size=3, activation='relu')

    # Sample input
    x = np.random.randn(2, 4)  # Batch of 2 samples

    # Forward pass
    output = layer.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")

    # Backward pass (dummy gradient)
    d_output = np.random.randn(2, 3)
    d_input = layer.backward(d_output)
    print(f"Gradient w.r.t. input shape: {d_input.shape}")
