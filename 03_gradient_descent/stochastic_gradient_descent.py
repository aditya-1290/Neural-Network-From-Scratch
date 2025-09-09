import numpy as np
import matplotlib.pyplot as plt

def stochastic_gradient_descent(X, y, f, df, w0, learning_rate=0.01, batch_size=32, max_iter=1000, tol=1e-6):
    """
    Stochastic Gradient Descent with mini-batches.

    Args:
        X (np.array): Feature matrix (n_samples, n_features)
        y (np.array): Target values
        f (function): Loss function
        df (function): Gradient of loss function
        w0 (np.array): Initial weights
        learning_rate (float): Step size
        batch_size (int): Size of mini-batch
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence

    Returns:
        np.array: Optimized weights
        list: History of weights
        list: History of loss values
    """
    w = w0.copy()
    n_samples = X.shape[0]
    w_history = [w.copy()]
    loss_history = []

    for epoch in range(max_iter):
        # Shuffle data
        indices = np.random.permutation(n_samples)

        epoch_loss = 0
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Compute gradient for batch
            grad = df(w, X_batch, y_batch)
            w_new = w - learning_rate * grad

            # Check for convergence (simplified)
            if np.linalg.norm(w_new - w) < tol:
                break

            w = w_new
            batch_loss = f(w, X_batch, y_batch)
            epoch_loss += batch_loss

        w_history.append(w.copy())
        loss_history.append(epoch_loss / (n_samples // batch_size))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss_history[-1]:.4f}")

    return w, w_history, loss_history

# Example: Linear regression with SGD
def mse_loss(w, X, y):
    y_pred = X @ w
    return np.mean((y_pred - y)**2)

def mse_gradient(w, X, y):
    y_pred = X @ w
    return (2 / len(y)) * X.T @ (y_pred - y)

if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 1000, 2
    X = np.random.randn(n_samples, n_features)
    true_w = np.array([2.0, -1.5])
    y = X @ true_w + 0.1 * np.random.randn(n_samples)

    # Add bias term
    X = np.hstack([np.ones((n_samples, 1)), X])

    w0 = np.zeros(n_features + 1)

    w_opt, w_hist, loss_hist = stochastic_gradient_descent(
        X, y, mse_loss, mse_gradient, w0,
        learning_rate=0.01, batch_size=32, max_iter=200
    )

    print(f"True weights: {true_w}")
    print(f"Estimated weights: {w_opt[1:]}")
    print(f"Bias: {w_opt[0]}")

    # Plot loss history
    plt.plot(loss_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SGD Loss Convergence')
    plt.tight_layout()
    plt.savefig('sgd_loss_convergence.png')
    plt.show()
