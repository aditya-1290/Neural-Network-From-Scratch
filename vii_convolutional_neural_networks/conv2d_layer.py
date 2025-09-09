import numpy as np
from im2col import im2col, col2im

class Conv2DLayer:
    """
    2D Convolutional layer implementation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initialize the Conv2D layer.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolution kernel
            stride (int): Stride of the convolution
            padding (int): Padding added to input
        """
        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        # Initialize weights and bias
        self.weights = np.random.randn(out_channels, in_channels, self.kernel_h, self.kernel_w) * 0.01
        self.bias = np.zeros((out_channels, 1))

        # Gradients
        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)

        # Cache for backpropagation
        self.input_cache = None
        self.col_cache = None

    def forward(self, input_data):
        """
        Forward pass through the Conv2D layer.

        Args:
            input_data (np.array): Input of shape (batch_size, in_channels, H, W)

        Returns:
            np.array: Output of shape (batch_size, out_channels, H_out, W_out)
        """
        self.input_cache = input_data
        batch_size, C, H, W = input_data.shape

        # im2col for each sample in batch
        self.col_cache = []
        cols = []
        for b in range(batch_size):
            col = im2col(input_data[b], self.kernel_h, self.kernel_w, self.stride, self.padding)
            cols.append(col)
            self.col_cache.append(col)

        # Stack columns
        col_stacked = np.hstack(cols)  # Shape: (C*KH*KW, batch_size * H_out * W_out)

        # Reshape weights for matrix multiplication
        weights_reshaped = self.weights.reshape(self.out_channels, -1).T  # Shape: (C*KH*KW, out_channels)

        # Convolution as matrix multiplication
        out = np.dot(col_stacked.T, weights_reshaped).T  # Shape: (out_channels, batch_size * H_out * W_out)

        # Add bias
        out += self.bias

        # Reshape to output
        H_out = (H + 2 * self.padding - self.kernel_h) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_w) // self.stride + 1
        out = out.reshape(self.out_channels, batch_size, H_out, W_out).transpose(1, 0, 2, 3)

        return out

    def backward(self, d_output):
        """
        Backward pass through the Conv2D layer.

        Args:
            d_output (np.array): Gradient of loss w.r.t. output

        Returns:
            np.array: Gradient of loss w.r.t. input
        """
        batch_size, C_out, H_out, W_out = d_output.shape

        # Reshape d_output
        d_output_reshaped = d_output.transpose(1, 0, 2, 3).reshape(C_out, -1)  # Shape: (out_channels, batch_size * H_out * W_out)

        # Gradient w.r.t. weights
        weights_reshaped = self.weights.reshape(self.out_channels, -1)
        col_stacked = np.hstack(self.col_cache)
        self.d_weights = np.dot(d_output_reshaped, col_stacked.T).reshape(self.weights.shape)

        # Gradient w.r.t. bias
        self.d_bias = np.sum(d_output_reshaped, axis=1, keepdims=True)

        # Gradient w.r.t. input
        d_col = np.dot(weights_reshaped.T, d_output_reshaped)  # Shape: (C*KH*KW, batch_size * H_out * W_out)

        # Split and col2im for each sample
        d_input = np.zeros_like(self.input_cache)
        col_size = d_col.shape[1] // batch_size
        for b in range(batch_size):
            d_col_b = d_col[:, b * col_size:(b + 1) * col_size]
            d_input[b] = col2im(d_col_b, self.input_cache[b].shape, self.kernel_h, self.kernel_w, self.stride, self.padding)

        return d_input

    def update(self, learning_rate):
        """
        Update weights and bias.

        Args:
            learning_rate (float): Learning rate
        """
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias

# Example usage
if __name__ == "__main__":
    # Sample input: batch of 2, 3 channels, 5x5
    input_data = np.random.randn(2, 3, 5, 5)
    print("Input shape:", input_data.shape)

    # Create Conv2D layer: 3 -> 16 channels, 3x3 kernel
    conv_layer = Conv2DLayer(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

    # Forward pass
    output = conv_layer.forward(input_data)
    print("Output shape:", output.shape)
    # Expected: (2, 16, 5, 5) since padding=1, stride=1

    # Backward pass (dummy gradient)
    d_output = np.random.randn(*output.shape)
    d_input = conv_layer.backward(d_output)
    print("Gradient w.r.t. input shape:", d_input.shape)

    # Update
    conv_layer.update(0.01)
    print("Weights updated successfully")
