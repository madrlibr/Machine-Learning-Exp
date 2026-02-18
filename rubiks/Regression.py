import pandas as pd
import numpy as np
from sympy import symbols, init_printing, latex
from IPython.display import display, Math

init_printing(use_latex='mathjax') 


class LogisticRegression:
    def __init__(self, x, y):
        self.bias = 0.0
        self.weight = 0.0
        self.x = np.array(x).reshape(-1, 1)
        self.y = np.array(y).reshape(-1, 1)
        
    def train(self, epoch, lr, tol):
        m = len(self.x)
        self.P = lambda za: 1 / (1 + np.exp(-za))
        lossB = np.inf
        for i in range(epoch):
            z = self.weight * self.x + self.bias
            yPred = self.P(z)

            #lossN = -(1/m) * np.sum(self.y * np.log(yPred) + (1 - self.y) * np.log(1 - yPred))

            """if abs(lossB - lossN) < tol:
                print(f"Convergen at iteration-{i}")
                break"""
            
            #lossB = lossN
            error = yPred - self.y
            
            dw = np.sum(self.x * error) / m
            db = np.sum(error) / m

            self.weight -= lr * dw
            self.bias -= lr * db

        return self.weight, self.bias

    def predict(self, x):
        P = self.P(za=self.weight * x + self.bias) 
        return P


class LinearRegression:
    def __init__(self, x, y):
        self.bias = 0
        self.weight = 0
        self.x = np.array(x).reshape(-1, 1)
        self.y = np.array(y).reshape(-1, 1)
        
    def train(self, epoch, lr):
        self.P = lambda w, b, x: (w * x) + b
        n = len(self.x)
        for i in range(epoch):
            yPred = self.P(self.weight, self.bias, self.x)
            error = yPred - self.y
            
            dw = (2/n) * np.sum(error * self.x)
            db = (2/n) * np.sum(error)
            
            self.weight -= lr * dw
            self.bias -= lr * db
            mse = np.mean(error**2)

        return self.weight, self.bias
    
    def predict(self, x):
        return self.P(self.weight, self.bias, x)

    def showFormula(self):
        print("Model:")
        display(Math(f'f(x) = {self.weight:.4f} \cdot x + {self.bias:.4f}'))
        print("Formulas:")
        display(Math(r'f(x) = w \cdot x + b'))
        display(Math(r'dw = \frac{2}{n} \sum_{i=1}^{n} (y_{pred} - y_i) \cdot x_i'))
        display(Math(r'db = \frac{2}{n} \sum_{i=1}^{n} (y_{pred} - y_i)'))
        

def MSE(yPred, yTrue):
    error = yPred - yTrue
    print(f"{yPred} - {yTrue} = {error}")
    print(f"Error^2 = {error ** 2}")
    return np.mean((yPred - yTrue) ** 2)

