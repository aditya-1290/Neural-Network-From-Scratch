import numpy as np
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_networks_from_scratch import Conv2DLayer
from neural_networks_from_scratch import MaxPoolingLayer
from neural_networks_from_scratch import DenseLayer

class LeNet5:
    """
    LeNet-5 Convolutional Neural Network implementation.
    """

    def __init__(self):
        """
        Initialize LeNet-5 architecture.
        """
        # Convolutional layers
        self.conv1 = Conv2DLayer(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = MaxPoolingLayer(kernel_size=2, stride=2)

        self.conv2 = Conv2DLayer(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = MaxPoolingLayer(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = DenseLayer(input_size=16*5*5, output_size=120, activation='tanh')
        self.fc2 = DenseLayer(input_size=120, output_size=84, activation='tanh')
        self.fc3 = DenseLayer(input_size=84, output_size=10, activation='linear')  # No activation for output

    def forward(self, x):
        """
        Forward pass through LeNet-5.

        Args:
            x (np.array): Input of shape (batch_size, 1, 32, 32)

        Returns:
            np.array: Output logits of shape (batch_size, 10)
        """
        # Conv1 -> Pool1
        x = self.conv1.forward(x)
        x = np.tanh(x)  # Tanh activation
        x = self.pool1.forward(x)

        # Conv2 -> Pool2
        x = self.conv2.forward(x)
        x = np.tanh(x)
        x = self.pool2.forward(x)

        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # FC layers
        x = self.fc1.forward(x)
        x = self.fc2.forward(x)
        x = self.fc3.forward(x)

        return x

    def backward(self, d_output):
        """
        Backward pass through LeNet-5.

        Args:
            d_output (np.array): Gradient of loss w.r.t. output

        Returns:
            np.array: Gradient of loss w.r.t. input
        """
        # FC layers backward
        d_output = self.fc3.backward(d_output)
        d_output = self.fc2.backward(d_output)
        d_output = self.fc1.backward(d_output)

        # Reshape for conv layers
        batch_size = d_output.shape[0]
        d_output = d_output.reshape(batch_size, 16, 5, 5)

        # Conv2 -> Pool2 backward
        d_output = self.pool2.backward(d_output)
        d_output = self.conv2.backward(d_output * (1 - np.tanh(self.conv2.input_cache)**2))  # Derivative of tanh

        # Conv1 -> Pool1 backward
        d_output = self.pool1.backward(d_output)
        d_output = self.conv1.backward(d_output * (1 - np.tanh(self.conv1.input_cache)**2))

        return d_output

    def update(self, learning_rate):
        """
        Update all parameters.

        Args:
            learning_rate (float): Learning rate
        """
        self.conv1.update(learning_rate)
        self.conv2.update(learning_rate)
        self.fc1.update(learning_rate)
        self.fc2.update(learning_rate)
        self.fc3.update(learning_rate)

    def predict(self, x):
        """
        Make predictions.

        Args:
            x (np.array): Input

        Returns:
            np.array: Predicted class indices
        """
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

# Example usage
if __name__ == "__main__":
    # Create LeNet-5
    lenet = LeNet5()

    # Sample input: batch of 2, 1 channel, 32x32
    x = np.random.randn(2, 1, 32, 32)

    # Forward pass
    output = lenet.forward(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output:", output)

    # Backward pass (dummy gradient)
    d_output = np.random.randn(2, 10)
    d_input = lenet.backward(d_output)
    print("Gradient w.r.t. input shape:", d_input.shape)

    # Update
    lenet.update(0.01)
    print("Parameters updated successfully")

    # Prediction
    predictions = lenet.predict(x)
    print("Predictions:", predictions)
