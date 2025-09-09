import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_networks_from_scratch import NeuralNetwork
from neural_networks_from_scratch import binary_cross_entropy, bce_derivative

# Generate XOR data
def generate_xor_data(n_samples=200):
    """
    Generate XOR dataset.

    Returns:
        X (np.array): Input features
        y (np.array): Target labels
    """
    # Generate random points in 2D
    X = np.random.randn(n_samples, 2)

    # XOR labels: 1 if points are in different quadrants, 0 otherwise
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    return X, y.reshape(-1, 1)

def train_binary_classifier():
    """
    Train a neural network for binary classification on XOR data.
    """
    # Generate data
    X, y = generate_xor_data()
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Create neural network: 2 inputs -> 4 hidden -> 1 output
    nn = NeuralNetwork([2, 4, 1], ['tanh', 'sigmoid'])

    # Training parameters
    learning_rate = 0.1
    epochs = 10000

    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        y_pred = nn.forward(X)

        # Compute loss
        loss = binary_cross_entropy(y, y_pred)
        losses.append(loss)

        # Backward pass
        d_loss = bce_derivative(y, y_pred)
        nn.backward(d_loss)

        # Update parameters
        nn.update(learning_rate)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print(f"Final loss: {losses[-1]:.4f}")

    # Plot decision boundary
    plot_decision_boundary(nn, X, y)

    return nn, losses

def plot_decision_boundary(nn, X, y):
    """
    Plot the decision boundary of the trained network.
    """
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict on mesh grid
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('XOR Classification Decision Boundary')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    train_binary_classifier()
