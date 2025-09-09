import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, df, x0, learning_rate=0.01, max_iter=1000, tol=1e-6):
    """
    Vanilla Gradient Descent optimization.

    Args:
        f (function): Objective function to minimize
        df (function): Gradient of the objective function
        x0 (np.array): Initial point
        learning_rate (float): Step size
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence

    Returns:
        np.array: Optimized point
        list: History of points
        list: History of function values
    """
    x = x0.copy()
    x_history = [x.copy()]
    f_history = [f(x)]

    for i in range(max_iter):
        grad = df(x)
        x_new = x - learning_rate * grad

        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        x_history.append(x.copy())
        f_history.append(f(x))

    return x, x_history, f_history

# Example: Minimize a simple quadratic function f(x) = x^2
def f(x):
    return x**2

def df(x):
    return 2 * x

if __name__ == "__main__":
    x0 = np.array([5.0])  # Starting point
    x_opt, x_hist, f_hist = gradient_descent(f, df, x0, learning_rate=0.1, max_iter=50)

    print(f"Optimal point: {x_opt}")
    print(f"Optimal value: {f(x_opt)}")

    # Plot the optimization path
    x_vals = np.linspace(-6, 6, 100)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_vals, f(x_vals), 'b-', label='f(x) = x²')
    plt.plot(x_hist, f_hist, 'ro-', markersize=3, label='GD path')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent Path')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(f_hist, 'g-', label='Function value')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.title('Convergence')
    plt.legend()

    plt.tight_layout()
    plt.show()
