import numpy as np
import matplotlib.pyplot as plt

def sigmoid_derivative(x):
    """
    Derivative of sigmoid function.
    """
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)

def tanh_derivative(x):
    """
    Derivative of tanh function.
    """
    return 1 - np.tanh(x)**2

def relu_derivative(x):
    """
    Derivative of ReLU function.
    """
    return np.where(x > 0, 1, 0)

def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of Leaky ReLU function.
    """
    return np.where(x > 0, 1, alpha)

def softmax_derivative(x):
    """
    Derivative of softmax function.
    Note: Softmax derivative is more complex and depends on the context.
    This is a simplified version for a single output.
    """
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)  # Approximation using sigmoid derivative

# Plotting the derivatives
if __name__ == "__main__":
    x = np.linspace(-10, 10, 400)

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 2, 1)
    plt.plot(x, sigmoid_derivative(x))
    plt.title("Sigmoid Derivative")

    plt.subplot(3, 2, 2)
    plt.plot(x, tanh_derivative(x))
    plt.title("Tanh Derivative")

    plt.subplot(3, 2, 3)
    plt.plot(x, relu_derivative(x))
    plt.title("ReLU Derivative")

    plt.subplot(3, 2, 4)
    plt.plot(x, leaky_relu_derivative(x))
    plt.title("Leaky ReLU Derivative")

    plt.subplot(3, 2, 5)
    plt.plot(x, softmax_derivative(x))
    plt.title("Softmax Derivative (approx)")

    plt.tight_layout()
    plt.show()

    # Also plot activation and derivative together for comparison
    plt.figure(figsize=(12, 4))

    for i, (name, func, deriv) in enumerate([
        ("Sigmoid", lambda x: 1/(1+np.exp(-x)), sigmoid_derivative),
        ("Tanh", np.tanh, tanh_derivative),
        ("ReLU", lambda x: np.maximum(0, x), relu_derivative)
    ]):
        plt.subplot(1, 3, i+1)
        plt.plot(x, func(x), label='Activation')
        plt.plot(x, deriv(x), label='Derivative')
        plt.title(f"{name} and its Derivative")
        plt.legend()

    plt.tight_layout()
    plt.show()
