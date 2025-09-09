import numpy as np
import matplotlib.pyplot as plt

def rmsprop(f, df, x0, learning_rate=0.01, beta=0.9, epsilon=1e-8, max_iter=1000, tol=1e-6):
    """
    RMSProp optimization algorithm.

    Args:
        f (function): Objective function to minimize
        df (function): Gradient of the objective function
        x0 (np.array): Initial point
        learning_rate (float): Step size
        beta (float): Decay rate for moving average of squared gradients
        epsilon (float): Small value to avoid division by zero
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence

    Returns:
        np.array: Optimized point
        list: History of points
        list: History of function values
    """
    x = x0.copy()
    v = np.zeros_like(x)  # Moving average of squared gradients
    x_history = [x.copy()]
    f_history = [f(x)]

    for i in range(max_iter):
        grad = df(x)
        v = beta * v + (1 - beta) * grad**2
        x_new = x - learning_rate * grad / (np.sqrt(v) + epsilon)

        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        x_history.append(x.copy())
        f_history.append(f(x))

    return x, x_history, f_history

# Example: Minimize a quadratic function with RMSProp
def f(x):
    return x**2

def df(x):
    return 2 * x

if __name__ == "__main__":
    x0 = np.array([5.0])

    x_opt, x_hist, f_hist = rmsprop(f, df, x0, learning_rate=0.1, beta=0.9, max_iter=50)

    print(f"Optimal point: {x_opt}")
    print(f"Optimal value: {f(x_opt)}")

    # Plot the optimization path
    x_vals = np.linspace(-6, 6, 100)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_vals, f(x_vals), 'b-', label='f(x) = x²')
    plt.plot(x_hist, f_hist, 'ro-', markersize=3, label='RMSProp path')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('RMSProp Optimization Path')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(f_hist, 'g-', label='Function value')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.title('RMSProp Convergence')
    plt.legend()

    plt.tight_layout()
    plt.savefig('rmsprop_optimization.png')
    plt.show()
