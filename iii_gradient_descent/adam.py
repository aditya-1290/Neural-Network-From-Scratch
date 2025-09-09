import numpy as np
import matplotlib.pyplot as plt

def adam(f, df, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000, tol=1e-6):
    """
    Adam optimization algorithm.

    Args:
        f (function): Objective function to minimize
        df (function): Gradient of the objective function
        x0 (np.array): Initial point
        learning_rate (float): Step size
        beta1 (float): Exponential decay rate for first moment
        beta2 (float): Exponential decay rate for second moment
        epsilon (float): Small value to avoid division by zero
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence

    Returns:
        np.array: Optimized point
        list: History of points
        list: History of function values
    """
    x = x0.copy()
    m = np.zeros_like(x)  # First moment
    v = np.zeros_like(x)  # Second moment
    t = 0  # Time step
    x_history = [x.copy()]
    f_history = [f(x)]

    for i in range(max_iter):
        t += 1
        grad = df(x)

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad

        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * grad**2

        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1**t)

        # Compute bias-corrected second moment estimate
        v_hat = v / (1 - beta2**t)

        # Update parameters
        x_new = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        x_history.append(x.copy())
        f_history.append(f(x))

    return x, x_history, f_history

# Example: Minimize a quadratic function with Adam
def f(x):
    return x**2

def df(x):
    return 2 * x

if __name__ == "__main__":
    x0 = np.array([5.0])

    x_opt, x_hist, f_hist = adam(f, df, x0, learning_rate=0.1, max_iter=50)

    print(f"Optimal point: {x_opt}")
    print(f"Optimal value: {f(x_opt)}")

    # Plot the optimization path
    x_vals = np.linspace(-6, 6, 100)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_vals, f(x_vals), 'b-', label='f(x) = x²')
    plt.plot(x_hist, f_hist, 'ro-', markersize=3, label='Adam path')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Adam Optimization Path')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(f_hist, 'g-', label='Function value')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.title('Adam Convergence')
    plt.legend()

    plt.tight_layout()
    plt.savefig('adam_optimization.png')
    plt.show()
