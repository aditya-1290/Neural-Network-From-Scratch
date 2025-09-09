import numpy as np
from neural_network import NeuralNetwork

def backward_pass_example():
    """
    Example of a complete backward pass through a neural network.
    """
    # Create a simple network
    nn = NeuralNetwork([3, 5, 2], ['relu', 'sigmoid'])

    # Sample input and target
    X = np.array([[0.1, 0.5, 0.8],
                  [0.2, 0.3, 0.9]])
    y_true = np.array([[1, 0],
                       [0, 1]])

    # Forward pass
    y_pred = nn.forward(X)

    print("Forward pass output:")
    print(y_pred)

    # Compute loss gradient (for demonstration, using MSE)
    loss_grad = 2 * (y_pred - y_true) / y_true.shape[0]

    print("\nLoss gradient:")
    print(loss_grad)

    # Backward pass
    nn.backward(loss_grad)

    print("\nGradients computed for all layers:")
    for i, layer in enumerate(nn.layers):
        print(f"Layer {i+1} weight gradient shape: {layer.d_weights.shape}")
        print(f"Layer {i+1} bias gradient shape: {layer.d_bias.shape}")

    # Update parameters
    learning_rate = 0.01
    nn.update(learning_rate)

    print(f"\nParameters updated with learning rate: {learning_rate}")

    # Verify the update
    print("\nWeight changes (first few elements):")
    for i, layer in enumerate(nn.layers):
        print(f"Layer {i+1}: {layer.weights[0, :3]}")

if __name__ == "__main__":
    backward_pass_example()
