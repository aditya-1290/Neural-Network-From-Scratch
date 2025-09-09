import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_networks_from_scratch import NeuralNetwork
from neural_networks_from_scratch import categorical_cross_entropy, cce_derivative

def load_and_preprocess_iris():
    """
    Load and preprocess the Iris dataset.

    Returns:
        X_train, X_test, y_train, y_test: Preprocessed data
    """
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # One-hot encode labels
    y_train_onehot = np.eye(3)[y_train]
    y_test_onehot = np.eye(3)[y_test]

    return X_train, X_test, y_train_onehot, y_test_onehot, y_train, y_test

def train_multi_class_classifier():
    """
    Train a neural network for multi-class classification on Iris dataset.
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = load_and_preprocess_iris()
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Create neural network: 4 inputs -> 10 hidden -> 3 outputs
    nn = NeuralNetwork([4, 10, 3], ['relu', 'softmax'])

    # Training parameters
    learning_rate = 0.01
    epochs = 1000

    # Training loop
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        # Forward pass
        y_pred_train = nn.forward(X_train)

        # Compute loss
        loss = categorical_cross_entropy(y_train, y_pred_train)
        train_losses.append(loss)

        # Backward pass
        d_loss = cce_derivative(y_train, y_pred_train)
        nn.backward(d_loss)

        # Update parameters
        nn.update(learning_rate)

        # Calculate accuracy
        train_accuracy = calculate_accuracy(y_pred_train, y_train_labels)
        train_accuracies.append(train_accuracy)

        # Test on validation set
        y_pred_test = nn.forward(X_test)
        test_loss = categorical_cross_entropy(y_test, y_pred_test)
        test_losses.append(test_loss)
        test_accuracy = calculate_accuracy(y_pred_test, y_test_labels)
        test_accuracies.append(test_accuracy)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

    print(f"Final Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}")
    print(f"Final Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}")

    # Plot results
    plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies)

    return nn

def calculate_accuracy(y_pred, y_true):
    """
    Calculate classification accuracy.

    Args:
        y_pred (np.array): Predicted probabilities
        y_true (np.array): True labels

    Returns:
        float: Accuracy
    """
    y_pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(y_pred_labels == y_true)

def plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies):
    """
    Plot training history.
    """
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_multi_class_classifier()
