import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """
    Hyperbolic tangent activation function.
    """
    return np.tanh(x)

def relu(x):
    """
    Rectified Linear Unit activation function.
    """
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    """
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    """
    Softmax activation function for multi-class classification.
    Numerically stable implementation.
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=-1, keepdims=True)

# Example usage and plotting
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = np.linspace(-10, 10, 400)

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 2, 1)
    plt.plot(x, sigmoid(x))
    plt.title("Sigmoid")

    plt.subplot(3, 2, 2)
    plt.plot(x, tanh(x))
    plt.title("Tanh")

    plt.subplot(3, 2, 3)
    plt.plot(x, relu(x))
    plt.title("ReLU")

    plt.subplot(3, 2, 4)
    plt.plot(x, leaky_relu(x))
    plt.title("Leaky ReLU")

    plt.subplot(3, 2, 5)
    # Softmax is usually applied to vectors, so plot softmax of a 3-element vector varying x
    x_vec = np.array([x, x/2, x/3]).T
    softmax_vals = np.apply_along_axis(softmax, 1, x_vec)
    plt.plot(x, softmax_vals[:, 0], label='class 1')
    plt.plot(x, softmax_vals[:, 1], label='class 2')
    plt.plot(x, softmax_vals[:, 2], label='class 3')
    plt.title("Softmax")
    plt.legend()

    plt.tight_layout()
    plt.show()
