import numpy as np
from dense_layer import DenseLayer

class NeuralNetwork:
    """
    A simple neural network class that combines multiple layers.
    """

    def __init__(self, layer_sizes, activations):
        """
        Initialize the neural network.

        Args:
            layer_sizes (list): List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activations (list): List of activation functions for each layer
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X (np.array): Input data

        Returns:
            np.array: Network output
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, d_output):
        """
        Backward pass through the network.

        Args:
            d_output (np.array): Gradient of loss with respect to output

        Returns:
            np.array: Gradient of loss with respect to input
        """
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)
        return d_output

    def update(self, learning_rate):
        """
        Update all layers' parameters.

        Args:
            learning_rate (float): Learning rate
        """
        for layer in self.layers:
            layer.update(learning_rate)

    def predict(self, X):
        """
        Make predictions (same as forward for regression/classification).

        Args:
            X (np.array): Input data

        Returns:
            np.array: Predictions
        """
        return self.forward(X)

# Example usage
if __name__ == "__main__":
    # Create a simple network: 4 inputs -> 10 hidden -> 3 outputs
    nn = NeuralNetwork([4, 10, 3], ['relu', 'linear'])

    # Sample input
    X = np.random.randn(5, 4)  # Batch of 5 samples

    # Forward pass
    output = nn.forward(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")

    # Backward pass (dummy gradient)
    d_output = np.random.randn(5, 3)
    nn.backward(d_output)

    # Update parameters
    nn.update(0.01)

    print("Network updated successfully")
