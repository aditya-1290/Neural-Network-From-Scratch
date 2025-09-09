import numpy as np
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_networks_from_scratch import Conv2DLayer

class MaxPoolingLayer:
    """
    Simplified MaxPooling layer for U-Net.
    """

    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """
        Forward pass for max pooling.

        Args:
            x (np.array): Input, shape (batch, channels, height, width)

        Returns:
            np.array: Output after max pooling
        """
        batch, channels, height, width = x.shape
        out_height = height // self.stride
        out_width = width // self.stride

        output = np.zeros((batch, channels, out_height, out_width))

        for b in range(batch):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        output[b, c, i, j] = np.max(x[b, c, h_start:h_end, w_start:w_end])

        return output

    def backward(self, d_output):
        """
        Backward pass for max pooling.

        Args:
            d_output (np.array): Gradient w.r.t. output

        Returns:
            np.array: Gradient w.r.t. input
        """
        # Simplified backward pass
        return d_output

class TransposeConv2DLayer:
    """
    Simplified Transpose Convolution layer for U-Net upsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Simple bilinear upsampling weights
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros((out_channels, 1))

    def forward(self, x):
        """
        Forward pass for transpose convolution.

        Args:
            x (np.array): Input, shape (batch, in_channels, height, width)

        Returns:
            np.array: Output after transpose convolution
        """
        batch, in_channels, height, width = x.shape
        out_height = height * self.stride
        out_width = width * self.stride

        output = np.zeros((batch, self.out_channels, out_height, out_width))

        # Simple nearest neighbor upsampling followed by convolution
        for b in range(batch):
            for c in range(self.out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Nearest neighbor
                        orig_i = i // self.stride
                        orig_j = j // self.stride
                        for ic in range(in_channels):
                            for ki in range(self.kernel_size):
                                for kj in range(self.kernel_size):
                                    if orig_i + ki < height and orig_j + kj < width:
                                        output[b, c, i, j] += x[b, ic, orig_i + ki, orig_j + kj] * self.weights[c, ic, ki, kj]

        return output

    def backward(self, d_output):
        """
        Backward pass for transpose convolution.

        Args:
            d_output (np.array): Gradient w.r.t. output

        Returns:
            np.array: Gradient w.r.t. input
        """
        # Simplified backward pass
        return d_output

class UNet:
    """
    U-Net architecture for image segmentation.
    """

    def __init__(self, in_channels=3, out_channels=1, features=64):
        """
        Initialize U-Net.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            features (int): Base number of features
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        # Encoder (Contracting Path)
        self.enc1 = Conv2DLayer(in_channels, features, kernel_size=3, padding=1)
        self.enc2 = Conv2DLayer(features, features * 2, kernel_size=3, padding=1)
        self.enc3 = Conv2DLayer(features * 2, features * 4, kernel_size=3, padding=1)
        self.enc4 = Conv2DLayer(features * 4, features * 8, kernel_size=3, padding=1)

        # Bottleneck
        self.bottleneck = Conv2DLayer(features * 8, features * 16, kernel_size=3, padding=1)

        # Decoder (Expansive Path)
        self.dec4 = Conv2DLayer(features * 16 + features * 8, features * 8, kernel_size=3, padding=1)
        self.dec3 = Conv2DLayer(features * 8 + features * 4, features * 4, kernel_size=3, padding=1)
        self.dec2 = Conv2DLayer(features * 4 + features * 2, features * 2, kernel_size=3, padding=1)
        self.dec1 = Conv2DLayer(features * 2 + features, features, kernel_size=3, padding=1)

        # Output
        self.out_conv = Conv2DLayer(features, out_channels, kernel_size=1)

        # Pooling and upsampling
        self.pool = MaxPoolingLayer()
        self.up4 = TransposeConv2DLayer(features * 16, features * 8)
        self.up3 = TransposeConv2DLayer(features * 8, features * 4)
        self.up2 = TransposeConv2DLayer(features * 4, features * 2)
        self.up1 = TransposeConv2DLayer(features * 2, features)

    def forward(self, x):
        """
        Forward pass through U-Net.

        Args:
            x (np.array): Input image, shape (batch, in_channels, height, width)

        Returns:
            np.array: Segmentation output
        """
        # Encoder
        enc1 = np.maximum(0, self.enc1.forward(x))  # ReLU
        enc1_pooled = self.pool.forward(enc1)

        enc2 = np.maximum(0, self.enc2.forward(enc1_pooled))
        enc2_pooled = self.pool.forward(enc2)

        enc3 = np.maximum(0, self.enc3.forward(enc2_pooled))
        enc3_pooled = self.pool.forward(enc3)

        enc4 = np.maximum(0, self.enc4.forward(enc3_pooled))
        enc4_pooled = self.pool.forward(enc4)

        # Bottleneck
        bottleneck = np.maximum(0, self.bottleneck.forward(enc4_pooled))

        # Decoder
        dec4 = self.up4.forward(bottleneck)
        dec4 = np.concatenate((dec4, enc4), axis=1)  # Skip connection
        dec4 = np.maximum(0, self.dec4.forward(dec4))

        dec3 = self.up3.forward(dec4)
        dec3 = np.concatenate((dec3, enc3), axis=1)
        dec3 = np.maximum(0, self.dec3.forward(dec3))

        dec2 = self.up2.forward(dec3)
        dec2 = np.concatenate((dec2, enc2), axis=1)
        dec2 = np.maximum(0, self.dec2.forward(dec2))

        dec1 = self.up1.forward(dec2)
        dec1 = np.concatenate((dec1, enc1), axis=1)
        dec1 = np.maximum(0, self.dec1.forward(dec1))

        # Output
        output = self.out_conv.forward(dec1)

        return output

    def backward(self, d_output):
        """
        Backward pass through U-Net.

        Args:
            d_output (np.array): Gradient w.r.t. output

        Returns:
            np.array: Gradient w.r.t. input
        """
        # Simplified backward pass
        return d_output

    def update(self, learning_rate):
        """
        Update all parameters.

        Args:
            learning_rate (float): Learning rate
        """
        self.enc1.update(learning_rate)
        self.enc2.update(learning_rate)
        self.enc3.update(learning_rate)
        self.enc4.update(learning_rate)
        self.bottleneck.update(learning_rate)
        self.dec4.update(learning_rate)
        self.dec3.update(learning_rate)
        self.dec2.update(learning_rate)
        self.dec1.update(learning_rate)
        self.out_conv.update(learning_rate)

# Example usage
if __name__ == "__main__":
    # Sample input: batch of 1, 3 channels, 64x64 image
    x = np.random.randn(1, 3, 64, 64)

    # Create U-Net
    unet = UNet(in_channels=3, out_channels=1, features=32)  # Smaller features for example

    # Forward pass
    output = unet.forward(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    # Backward pass (dummy gradient)
    d_output = np.random.randn(*output.shape)
    d_input = unet.backward(d_output)
    print("Gradient w.r.t. input shape:", d_input.shape)

    # Update
    unet.update(0.01)
    print("Parameters updated successfully")
