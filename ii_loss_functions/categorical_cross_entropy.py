import numpy as np

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Categorical Cross-Entropy loss for multi-class classification.

    Args:
        y_true (np.array): One-hot encoded true labels
        y_pred (np.array): Predicted probabilities (softmax outputs)
        epsilon (float): Small value to avoid log(0)

    Returns:
        float: Categorical cross-entropy loss
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cce_derivative(y_true, y_pred):
    """
    Derivative of Categorical Cross-Entropy with respect to predictions.

    Args:
        y_true (np.array): One-hot encoded true labels
        y_pred (np.array): Predicted probabilities

    Returns:
        np.array: Gradient of CCE
    """
    return - (y_true / y_pred) / y_true.shape[0]

# Example usage
if __name__ == "__main__":
    # Sample data for 3 classes and 4 samples
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    y_pred = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.6, 0.3, 0.1]
    ])

    loss = categorical_cross_entropy(y_true, y_pred)
    print(f"CCE Loss: {loss}")

    grad = cce_derivative(y_true, y_pred)
    print(f"Gradient:\n{grad}")
