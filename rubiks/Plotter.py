import numpy as np
import matplotlib.pyplot as plt

def LinearPlot(X, Y, weight, bias):
    plt.figure(figsize=(6, 3))
    yp = weight * X + bias
    plt.plot(X, yp, label=f'(weight={weight}, bias={bias})', color='red')
    plt.scatter(X, Y)
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def LogisticPlot(weight, bias):
    x = np.linspace(-10, 10, 1000)
    y = 1 / (1 + np.exp(-(weight * x + bias)))
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f'(weight={weight}, bias={bias})', color='blue', linewidth=2)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=-bias/weight, color='green', linestyle='--', alpha=0.5, label='Decision Boundary')
    plt.title('Sigmoid Curve')
    plt.xlabel('X')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()