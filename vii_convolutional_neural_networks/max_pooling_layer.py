import numpy as np
from im2col import im2col, col2im

class MaxPoolingLayer:
    """
    Max Pooling layer implementation.
    """

    def __init__(self, kernel_size, stride=None):
        """
        Initialize the MaxPooling layer.

        Args:
            kernel_size (int or tuple): Size of the pooling window
            stride (int or None): Stride of the pooling window. If None, defaults to kernel_size.
        """
        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size

        self.stride = stride if stride is not None else kernel_size

        # Cache for backpropagation
        self.input_cache = None
        self.argmax_cache = None

    def forward(self, input_data):
        """
        Forward pass through the MaxPooling layer.

        Args:
            input_data (np.array): Input of shape (batch_size, channels, H, W)

        Returns:
            np.array: Output after max pooling
        """
        self.input_cache = input_data
        batch_size, channels, H, W = input_data.shape

        # Output dimensions
        H_out = (H - self.kernel_h) // self.stride + 1
        W_out = (W - self.kernel_w) // self.stride + 1

        # Initialize output
        output = np.zeros((batch_size, channels, H_out, W_out))

        # Initialize argmax cache
        self.argmax_cache = np.zeros_like(output, dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                # Extract patches using im2col
                col = im2col(input_data[b, c:c+1], self.kernel_h, self.kernel_w, self.stride, padding=0)
                # Find max and argmax
                max_indices = np.argmax(col, axis=0)
                max_values = col[max_indices, range(col.shape[1])]
                output[b, c] = max_values.reshape(H_out, W_out)
                self.argmax_cache[b, c] = max_indices.reshape(H_out, W_out)

        return output

    def backward(self, d_output):
        """
        Backward pass through the MaxPooling layer.

        Args:
            d_output (np.array): Gradient of loss w.r.t. output

        Returns:
            np.array: Gradient of loss w.r.t. input
        """
        batch_size, channels, H_out, W_out = d_output.shape
        d_input = np.zeros_like(self.input_cache)

        for b in range(batch_size):
            for c in range(channels):
                # Extract patches using im2col
                col = im2col(self.input_cache[b, c:c+1], self.kernel_h, self.kernel_w, self.stride, padding=0)
                d_col = np.zeros_like(col)

                # Flatten d_output for this channel
                d_out_flat = d_output[b, c].flatten()

                # Assign gradients to max positions
                argmax_flat = self.argmax_cache[b, c].flatten()
                d_col[argmax_flat, range(d_col.shape[1])] = d_out_flat

                # Convert back to image shape
                d_input[b, c] = col2im(d_col, self.input_cache[b, c].shape, self.kernel_h, self.kernel_w, self.stride, padding=0)

        return d_input

# Example usage
if __name__ == "__main__":
    # Sample input: batch of 2, 3 channels, 4x4
    input_data = np.random.randn(2, 3, 4, 4)
    print("Input shape:", input_data.shape)

    # Create MaxPooling layer: 2x2 kernel, stride 2
    maxpool_layer = MaxPoolingLayer(kernel_size=2, stride=2)

    # Forward pass
    output = maxpool_layer.forward(input_data)
    print("Output shape:", output.shape)
    # Expected: (2, 3, 2, 2)

    # Backward pass (dummy gradient)
    d_output = np.random.randn(*output.shape)
    d_input = maxpool_layer.backward(d_output)
    print("Gradient w.r.t. input shape:", d_input.shape)
