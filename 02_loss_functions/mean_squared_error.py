import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error (MSE) loss function for regression tasks.

    Args:
        y_true (np.array): True target values
        y_pred (np.array): Predicted values

    Returns:
        float: Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    """
    Derivative of MSE with respect to predictions.

    Args:
        y_true (np.array): True target values
        y_pred (np.array): Predicted values

    Returns:
        np.array: Gradient of MSE
    """
    return 2 * (y_pred - y_true) / len(y_true)

# Example usage
if __name__ == "__main__":
    # Sample data
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    # Compute MSE
    mse = mean_squared_error(y_true, y_pred)
    print(f"MSE: {mse}")

    # Compute gradient
    grad = mse_derivative(y_true, y_pred)
    print(f"Gradient: {grad}")

    # Manual verification
    manual_mse = np.mean((y_true - y_pred) ** 2)
    print(f"Manual MSE: {manual_mse}")
    print(f"Match: {np.isclose(mse, manual_mse)}")
