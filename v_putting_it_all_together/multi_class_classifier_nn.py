import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_networks_from_scratch import NeuralNetwork
from neural_networks_from_scratch import categorical_cross_entropy, cce_derivative

def load_data():
    """
    Load and preprocess the Iris dataset for multi-class classification.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)

    # One-hot encode the target labels
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_multi_class_classifier():
    """
    Train a neural network for multi-class classification on Iris dataset.
    """
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    # Create neural network: 4 inputs -> 10 hidden -> 3 outputs
    nn = NeuralNetwork([4, 10, 3], ['relu', 'softmax'])

    # Training parameters
    learning_rate = 0.01
    epochs = 5000

    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        y_pred = nn.forward(X_train)

        # Compute loss
        loss = categorical_cross_entropy(y_train, y_pred)
        losses.append(loss)

        # Backward pass
        d_loss = cce_derivative(y_train, y_pred)
        nn.backward(d_loss)

        # Update parameters
        nn.update(learning_rate)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print(f"Final loss: {losses[-1]:.4f}")

    # Evaluate accuracy on test set
    y_test_pred = nn.predict(X_test)
    accuracy = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test, axis=1))
    print(f"Test accuracy: {accuracy:.4f}")

    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    return nn, losses

if __name__ == "__main__":
    train_multi_class_classifier()
