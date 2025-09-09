import numpy as np

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Binary Cross-Entropy loss for binary classification.

    Args:
        y_true (np.array): True binary labels (0 or 1)
        y_pred (np.array): Predicted probabilities (between 0 and 1)
        epsilon (float): Small value to avoid log(0)

    Returns:
        float: Binary cross-entropy loss
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute BCE
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

def bce_derivative(y_true, y_pred):
    """
    Derivative of BCE with respect to predictions.

    Args:
        y_true (np.array): True binary labels
        y_pred (np.array): Predicted probabilities

    Returns:
        np.array: Gradient of BCE
    """
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * len(y_true))

# Example usage
if __name__ == "__main__":
    # Sample data for binary classification
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2])

    # Compute BCE
    bce = binary_cross_entropy(y_true, y_pred)
    print(f"BCE: {bce}")

    # Compute gradient
    grad = bce_derivative(y_true, y_pred)
    print(f"Gradient: {grad}")

    # Manual verification
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    manual_bce = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    print(f"Manual BCE: {manual_bce}")
    print(f"Match: {np.isclose(bce, manual_bce)}")
