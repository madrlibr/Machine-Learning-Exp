import numpy as np
import matplotlib.pyplot as plt

def LinearPlot(X, Y, weight, bias):
    plt.figure(figsize=(6, 3))
    yp = weight * X + bias
    plt.plot(X, yp, color='red', label='line')
    plt.scatter(X, Y)
    plt.grid(True)
    plt.show()

def LogisticPlot(weight, bias):
    weight = round(float(weight, 3))
    bias = round(float(bias, 3))
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-(weight * x + bias)))
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f'(w={weight}, b={bias})', color='blue', linewidth=2)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=-bias/weight, color='green', linestyle='--', alpha=0.5, label='Decision Boundary')
    plt.title('Sigmoid Curve')
    plt.xlabel('X')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()