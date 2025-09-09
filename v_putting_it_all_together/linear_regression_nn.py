import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_networks_from_scratch.iv_building_a_network import NeuralNetwork
from neural_networks_from_scratch import mean_squared_error, mse_derivative

def generate_linear_data(n_samples=100, noise=0.1):
    """
    Generate synthetic linear regression data.

    Args:
        n_samples (int): Number of samples
        noise (float): Noise level

    Returns:
        X (np.array): Input features
        y (np.array): Target values
    """
    X = np.random.randn(n_samples, 1)
    y = 2 * X + 1 + noise * np.random.randn(n_samples, 1)
    return X, y

def train_linear_regression():
    """
    Train a neural network for linear regression.
    """
    # Generate data
    X, y = generate_linear_data()
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Create neural network: 1 input -> 1 output (no hidden layers)
    nn = NeuralNetwork([1, 1], ['linear'])

    # Training parameters
    learning_rate = 0.01
    epochs = 1000

    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        y_pred = nn.forward(X)

        # Compute loss
        loss = mean_squared_error(y, y_pred)
        losses.append(loss)

        # Backward pass
        d_loss = mse_derivative(y, y_pred)
        nn.backward(d_loss)

        # Update parameters
        nn.update(learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print(f"Final loss: {losses[-1]:.4f}")

    # Plot results
    plot_results(nn, X, y)

    return nn, losses

def plot_results(nn, X, y):
    """
    Plot the training data and model predictions.
    """
    # Predictions
    X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = nn.predict(X_test)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label='Training data', alpha=0.7)
    plt.plot(X_test, y_pred, 'r-', label='Model prediction', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression with Neural Network')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_linear_regression()
