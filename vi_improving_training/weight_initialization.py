import numpy as np

def zero_init(shape):
    """
    Zero weight initialization.

    Args:
        shape (tuple): Shape of the weight matrix

    Returns:
        np.array: Zero-initialized weights
    """
    return np.zeros(shape)

def random_init(shape, scale=0.01):
    """
    Random weight initialization with small values.

    Args:
        shape (tuple): Shape of the weight matrix
        scale (float): Scaling factor for random values

    Returns:
        np.array: Randomly initialized weights
    """
    return np.random.randn(*shape) * scale

def xavier_init(shape):
    """
    Xavier/Glorot weight initialization for tanh/sigmoid activations.

    Args:
        shape (tuple): Shape of the weight matrix (fan_in, fan_out)

    Returns:
        np.array: Xavier-initialized weights
    """
    fan_in, fan_out = shape
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

def he_init(shape):
    """
    He weight initialization for ReLU activations.

    Args:
        shape (tuple): Shape of the weight matrix (fan_in, fan_out)

    Returns:
        np.array: He-initialized weights
    """
    fan_in, fan_out = shape
    std = np.sqrt(2 / fan_in)
    return np.random.randn(*shape) * std

# Example usage
if __name__ == "__main__":
    shape = (10, 5)  # Example shape: 10 inputs, 5 outputs

    print("Zero initialization:")
    weights_zero = zero_init(shape)
    print(f"Shape: {weights_zero.shape}")
    print(f"Mean: {np.mean(weights_zero):.6f}, Std: {np.std(weights_zero):.6f}")
    print()

    print("Random initialization:")
    weights_random = random_init(shape)
    print(f"Shape: {weights_random.shape}")
    print(f"Mean: {np.mean(weights_random):.6f}, Std: {np.std(weights_random):.6f}")
    print()

    print("Xavier initialization:")
    weights_xavier = xavier_init(shape)
    print(f"Shape: {weights_xavier.shape}")
    print(f"Mean: {np.mean(weights_xavier):.6f}, Std: {np.std(weights_xavier):.6f}")
    print()

    print("He initialization:")
    weights_he = he_init(shape)
    print(f"Shape: {weights_he.shape}")
    print(f"Mean: {np.mean(weights_he):.6f}, Std: {np.std(weights_he):.6f}")
    print()

    # Plot distributions
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    weights_list = [weights_zero, weights_random, weights_xavier, weights_he]
    titles = ['Zero Init', 'Random Init', 'Xavier Init', 'He Init']

    for i, (weights, title) in enumerate(zip(weights_list, titles)):
        axes[i].hist(weights.flatten(), bins=50, alpha=0.7)
        axes[i].set_title(title)
        axes[i].set_xlabel('Weight Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('weight_initializations.png')
    plt.show()
