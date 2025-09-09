import numpy as np

def l1_penalty(weights, lambda_l1=0.01):
    """
    L1 regularization penalty (Lasso).

    Args:
        weights (np.array or list): Model weights
        lambda_l1 (float): Regularization strength

    Returns:
        float: L1 penalty value
    """
    if isinstance(weights, list):
        return lambda_l1 * sum(np.sum(np.abs(w)) for w in weights)
    else:
        return lambda_l1 * np.sum(np.abs(weights))

def l2_penalty(weights, lambda_l2=0.01):
    """
    L2 regularization penalty (Ridge).

    Args:
        weights (np.array or list): Model weights
        lambda_l2 (float): Regularization strength

    Returns:
        float: L2 penalty value
    """
    if isinstance(weights, list):
        return lambda_l2 * sum(np.sum(w**2) for w in weights)
    else:
        return lambda_l2 * np.sum(weights**2)

def l1_derivative(weights, lambda_l1=0.01):
    """
    Derivative of L1 penalty with respect to weights.

    Args:
        weights (np.array or list): Model weights
        lambda_l1 (float): Regularization strength

    Returns:
        np.array or list: Gradient of L1 penalty
    """
    if isinstance(weights, list):
        return [lambda_l1 * np.sign(w) for w in weights]
    else:
        return lambda_l1 * np.sign(weights)

def l2_derivative(weights, lambda_l2=0.01):
    """
    Derivative of L2 penalty with respect to weights.

    Args:
        weights (np.array or list): Model weights
        lambda_l2 (float): Regularization strength

    Returns:
        np.array or list: Gradient of L2 penalty
    """
    if isinstance(weights, list):
        return [2 * lambda_l2 * w for w in weights]
    else:
        return 2 * lambda_l2 * weights

def elastic_net_penalty(weights, lambda_l1=0.01, lambda_l2=0.01):
    """
    Elastic Net regularization (combination of L1 and L2).

    Args:
        weights (np.array or list): Model weights
        lambda_l1 (float): L1 regularization strength
        lambda_l2 (float): L2 regularization strength

    Returns:
        float: Elastic Net penalty value
    """
    return l1_penalty(weights, lambda_l1) + l2_penalty(weights, lambda_l2)

def elastic_net_derivative(weights, lambda_l1=0.01, lambda_l2=0.01):
    """
    Derivative of Elastic Net penalty with respect to weights.

    Args:
        weights (np.array or list): Model weights
        lambda_l1 (float): L1 regularization strength
        lambda_l2 (float): L2 regularization strength

    Returns:
        np.array or list: Gradient of Elastic Net penalty
    """
    l1_grad = l1_derivative(weights, lambda_l1)
    l2_grad = l2_derivative(weights, lambda_l2)

    if isinstance(weights, list):
        return [l1 + l2 for l1, l2 in zip(l1_grad, l2_grad)]
    else:
        return l1_grad + l2_grad

# Example usage
if __name__ == "__main__":
    # Sample weights
    weights = np.random.randn(10, 5)

    print("Sample weights shape:", weights.shape)
    print("Sample weights mean:", np.mean(weights))
    print("Sample weights std:", np.std(weights))
    print()

    # L1 regularization
    l1_pen = l1_penalty(weights)
    l1_grad = l1_derivative(weights)
    print("L1 penalty:", l1_pen)
    print("L1 gradient shape:", l1_grad.shape)
    print("L1 gradient mean:", np.mean(l1_grad))
    print()

    # L2 regularization
    l2_pen = l2_penalty(weights)
    l2_grad = l2_derivative(weights)
    print("L2 penalty:", l2_pen)
    print("L2 gradient shape:", l2_grad.shape)
    print("L2 gradient mean:", np.mean(l2_grad))
    print()

    # Elastic Net
    en_pen = elastic_net_penalty(weights)
    en_grad = elastic_net_derivative(weights)
    print("Elastic Net penalty:", en_pen)
    print("Elastic Net gradient shape:", en_grad.shape)
    print("Elastic Net gradient mean:", np.mean(en_grad))
    print()

    # Test with list of weights (e.g., multiple layers)
    weights_list = [np.random.randn(10, 5), np.random.randn(5, 3)]
    print("L1 penalty for list:", l1_penalty(weights_list))
    print("L2 penalty for list:", l2_penalty(weights_list))
    print("Elastic Net penalty for list:", elastic_net_penalty(weights_list))
    print()

    # Demonstrate effect on weight updates
    print("Effect on weight updates:")
    learning_rate = 0.01
    original_weights = weights.copy()

    # Without regularization
    grad_no_reg = np.random.randn(*weights.shape)  # Dummy gradient
    weights_no_reg = original_weights - learning_rate * grad_no_reg

    # With L2 regularization
    grad_l2 = grad_no_reg + l2_derivative(original_weights)
    weights_l2 = original_weights - learning_rate * grad_l2

    print("Original weights norm:", np.linalg.norm(original_weights))
    print("Weights without reg norm:", np.linalg.norm(weights_no_reg))
    print("Weights with L2 reg norm:", np.linalg.norm(weights_l2))
    print("L2 regularization reduces weight magnitude:", np.linalg.norm(weights_l2) < np.linalg.norm(weights_no_reg))
