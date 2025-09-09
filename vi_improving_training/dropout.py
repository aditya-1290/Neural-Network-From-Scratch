import numpy as np

class DropoutLayer:
    """
    Dropout layer implementation for regularization.
    """

    def __init__(self, dropout_rate=0.5):
        """
        Initialize the Dropout layer.

        Args:
            dropout_rate (float): Probability of dropping a neuron (0 < rate < 1)
        """
        self.dropout_rate = dropout_rate
        self.training = True
        self.mask = None

    def forward(self, input_data):
        """
        Forward pass through the Dropout layer.

        Args:
            input_data (np.array): Input data

        Returns:
            np.array: Output with dropout applied (if training)
        """
        if self.training:
            # Create dropout mask
            self.mask = np.random.rand(*input_data.shape) > self.dropout_rate
            # Apply dropout and scale
            output = input_data * self.mask / (1 - self.dropout_rate)
        else:
            # During inference, use all neurons (no scaling needed as per some implementations)
            output = input_data

        return output

    def backward(self, d_output):
        """
        Backward pass through the Dropout layer.

        Args:
            d_output (np.array): Gradient of loss with respect to output

        Returns:
            np.array: Gradient of loss with respect to input
        """
        if self.training:
            # Apply the same mask to the gradients
            d_input = d_output * self.mask / (1 - self.dropout_rate)
        else:
            d_input = d_output

        return d_input

    def set_training_mode(self, training=True):
        """
        Set the training mode.

        Args:
            training (bool): True for training, False for inference
        """
        self.training = training

# Example usage
if __name__ == "__main__":
    # Create a Dropout layer
    dropout_layer = DropoutLayer(dropout_rate=0.5)

    # Sample input
    x = np.random.randn(5, 4)  # Batch of 5 samples, 4 features

    print("Input shape:", x.shape)
    print("Input:")
    print(x)
    print()

    # Forward pass (training mode)
    output = dropout_layer.forward(x)
    print("Output shape:", output.shape)
    print("Output (training mode):")
    print(output)
    print("Mask:")
    print(dropout_layer.mask.astype(int))
    print()

    # Backward pass (dummy gradient)
    d_output = np.random.randn(5, 4)
    d_input = dropout_layer.backward(d_output)
    print("Gradient w.r.t. input shape:", d_input.shape)
    print("Gradient w.r.t. input (training mode):")
    print(d_input)
    print()

    # Inference mode
    dropout_layer.set_training_mode(False)
    output_inf = dropout_layer.forward(x)
    print("Output (inference mode):")
    print(output_inf)
    print("Are input and output equal in inference?", np.allclose(x, output_inf))
    print()

    # Test with different dropout rates
    print("Testing different dropout rates:")
    for rate in [0.2, 0.5, 0.8]:
        layer = DropoutLayer(dropout_rate=rate)
        out = layer.forward(x)
        kept_ratio = np.mean(layer.mask)
        print(f"Dropout rate {rate}: Kept ratio {kept_ratio:.3f}")
