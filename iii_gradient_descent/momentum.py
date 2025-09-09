import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_with_momentum(f, df, x0, learning_rate=0.01, momentum=0.9, max_iter=1000, tol=1e-6):
    """
    Gradient Descent with Momentum optimization.

    Args:
        f (function): Objective function to minimize
        df (function): Gradient of the objective function
        x0 (np.array): Initial point
        learning_rate (float): Step size
        momentum (float): Momentum coefficient (0 <= momentum < 1)
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence

    Returns:
        np.array: Optimized point
        list: History of points
        list: History of function values
    """
    x = x0.copy()
    v = np.zeros_like(x)  # Velocity
    x_history = [x.copy()]
    f_history = [f(x)]

    for i in range(max_iter):
        grad = df(x)
        v = momentum * v - learning_rate * grad
        x_new = x + v

        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        x_history.append(x.copy())
        f_history.append(f(x))

    return x, x_history, f_history

# Example: Minimize a quadratic function with momentum
def f(x):
    return x**2

def df(x):
    return 2 * x

if __name__ == "__main__":
    x0 = np.array([5.0])

    # Compare vanilla GD and GD with momentum
    x_opt_vanilla, x_hist_vanilla, f_hist_vanilla = gradient_descent_with_momentum(
        f, df, x0, learning_rate=0.1, momentum=0.0, max_iter=50
    )

    x_opt_momentum, x_hist_momentum, f_hist_momentum = gradient_descent_with_momentum(
        f, df, x0, learning_rate=0.1, momentum=0.9, max_iter=50
    )

    print(f"Vanilla GD - Optimal point: {x_opt_vanilla}, Value: {f(x_opt_vanilla)}")
    print(f"Momentum GD - Optimal point: {x_opt_momentum}, Value: {f(x_opt_momentum)}")

    # Plot comparison
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    x_vals = np.linspace(-6, 6, 100)
    plt.plot(x_vals, f(x_vals), 'b-', label='f(x) = x²')
    plt.plot(x_hist_vanilla, f_hist_vanilla, 'ro-', markersize=3, label='Vanilla GD')
    plt.plot(x_hist_momentum, f_hist_momentum, 'go-', markersize=3, label='Momentum GD')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Optimization Paths')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(f_hist_vanilla, 'r-', label='Vanilla GD')
    plt.plot(f_hist_momentum, 'g-', label='Momentum GD')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.title('Convergence Comparison')
    plt.legend()

    plt.tight_layout()
    plt.savefig('momentum_comparison.png')
    plt.show()
