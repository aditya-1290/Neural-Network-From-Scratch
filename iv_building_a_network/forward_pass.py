import numpy as np
from neural_network import NeuralNetwork

def forward_pass_example():
    """
    Example of a complete forward pass through a neural network.
    """
    # Create a simple network
    nn = NeuralNetwork([3, 5, 2], ['relu', 'sigmoid'])

    # Sample input
    X = np.array([[0.1, 0.5, 0.8],
                  [0.2, 0.3, 0.9]])

    print("Input:")
    print(X)
    print(f"Input shape: {X.shape}")

    # Forward pass
    output = nn.forward(X)

    print("\nNetwork Architecture:")
    print("Layer 1: 3 -> 5 (ReLU)")
    print("Layer 2: 5 -> 2 (Sigmoid)")

    print("\nOutput:")
    print(output)
    print(f"Output shape: {output.shape}")

    # Show intermediate activations
    print("\nIntermediate activations:")
    for i, layer in enumerate(nn.layers):
        print(f"Layer {i+1} output shape: {layer.output_cache.shape}")

if __name__ == "__main__":
    forward_pass_example()
