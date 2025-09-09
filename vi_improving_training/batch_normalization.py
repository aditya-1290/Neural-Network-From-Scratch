import numpy as np

class BatchNormLayer:
    """
    Batch Normalization layer implementation.
    """

    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        """
        Initialize the BatchNorm layer.

        Args:
            num_features (int): Number of features to normalize
            momentum (float): Momentum for running statistics
            eps (float): Small value for numerical stability
        """
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Trainable parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))

        # Gradients
        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.beta)

        # Running statistics for inference
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        # Cache for backpropagation
        self.input_cache = None
        self.x_hat_cache = None
        self.mean_cache = None
        self.var_cache = None
        self.std_cache = None

        # Training mode flag
        self.training = True

    def forward(self, input_data):
        """
        Forward pass through the BatchNorm layer.

        Args:
            input_data (np.array): Input data of shape (batch_size, num_features)

        Returns:
            np.array: Normalized output
        """
        self.input_cache = input_data

        if self.training:
            # Compute batch statistics
            self.mean_cache = np.mean(input_data, axis=0, keepdims=True)
            self.var_cache = np.var(input_data, axis=0, keepdims=True)
            self.std_cache = np.sqrt(self.var_cache + self.eps)

            # Normalize
            self.x_hat_cache = (input_data - self.mean_cache) / self.std_cache

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean_cache
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var_cache
        else:
            # Use running statistics for inference
            self.x_hat_cache = (input_data - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # Scale and shift
        output = self.gamma * self.x_hat_cache + self.beta
        return output

    def backward(self, d_output):
        """
        Backward pass through the BatchNorm layer.

        Args:
            d_output (np.array): Gradient of loss with respect to output

        Returns:
            np.array: Gradient of loss with respect to input
        """
        batch_size = d_output.shape[0]

        # Gradient w.r.t. gamma and beta
        self.d_gamma = np.sum(d_output * self.x_hat_cache, axis=0, keepdims=True)
        self.d_beta = np.sum(d_output, axis=0, keepdims=True)

        # Gradient w.r.t. normalized input
        d_x_hat = d_output * self.gamma

        # Gradient w.r.t. input
        d_input = (batch_size * d_x_hat - np.sum(d_x_hat, axis=0, keepdims=True) -
                   self.x_hat_cache * np.sum(d_x_hat * self.x_hat_cache, axis=0, keepdims=True)) / (batch_size * self.std_cache)

        return d_input

    def update(self, learning_rate):
        """
        Update gamma and beta parameters.

        Args:
            learning_rate (float): Learning rate
        """
        self.gamma -= learning_rate * self.d_gamma
        self.beta -= learning_rate * self.d_beta

    def set_training_mode(self, training=True):
        """
        Set the training mode.

        Args:
            training (bool): True for training, False for inference
        """
        self.training = training

# Example usage
if __name__ == "__main__":
    # Create a BatchNorm layer
    bn_layer = BatchNormLayer(num_features=3)

    # Sample input
    x = np.random.randn(10, 3)  # Batch of 10 samples, 3 features

    print("Input shape:", x.shape)
    print("Input mean:", np.mean(x, axis=0))
    print("Input std:", np.std(x, axis=0))
    print()

    # Forward pass (training mode)
    output = bn_layer.forward(x)
    print("Output shape:", output.shape)
    print("Output mean:", np.mean(output, axis=0))
    print("Output std:", np.std(output, axis=0))
    print()

    # Backward pass (dummy gradient)
    d_output = np.random.randn(10, 3)
    d_input = bn_layer.backward(d_output)
    print("Gradient w.r.t. input shape:", d_input.shape)
    print()

    # Update parameters
    bn_layer.update(0.01)
    print("Gamma:", bn_layer.gamma)
    print("Beta:", bn_layer.beta)
    print()

    # Inference mode
    bn_layer.set_training_mode(False)
    x_test = np.random.randn(5, 3)
    output_test = bn_layer.forward(x_test)
    print("Inference output mean:", np.mean(output_test, axis=0))
    print("Inference output std:", np.std(output_test, axis=0))
