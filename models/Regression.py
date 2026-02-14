import math
import numpy as np

#The logistic regression still not fixed yet
"""class LogisticRegression:
    def __init__(self, x, y, lr):
        self.b = 0
        self.w = 0
        self.x = x
        self.y = x
        self.lr = lr

    def gradientDescent(self):
        P = lambda z: 1 / (1 + (math.e ** -z))
        epoch = 10
        for i in range(epoch):
            for X, Y in zip(self.x, self.y):
                print(f"w: {self.w}")
                print(f"b: {self.b}")
                self.z = self.b + (self.w * X)
                self.yp = P(self.z)

                error = (self.yp - Y)
                self.lossb = error
                self.lossW = error * X

                self.w = self.w - (self.lr * self.lossW)
                self.b = self.b - (self.lr * self.lossb)
                
                self.z = self.b + (self.w * X)
                self.yp = P(self.z)
                
        self.z = self.b + (self.w * self.x)
        self.yp = P(self.z)
        
        return self.yp, self.w, self.b
    
    def predict(self, x):
        self.z = self.b + (self.w * x)
        P = 1 / (1 + math.e ** -self.z)

        return round(P, 3)"""

class LinearRegression:
    def __init__(self, x, y, lr):
        self.bias = 0
        self.weight = 0
        self.x = self.x = np.array(x).reshape(-1, 1)
        self.y = self.y = np.array(y).reshape(-1, 1)
        self.lr = lr
        
    def fit(self):
        epoch = 1000
        n = len(self.x)
        for i in range(epoch):
            yPred = self.weight * self.x + self.bias
            error = yPred - self.y
            
            dw = (2/n) * np.sum(error * self.x)
            db = (2/n) * np.sum(error)
            
            self.weight -= self.lr * dw
            self.bias -= self.lr * db
            
        return self.weight, self.bias
    
    def predict(self, x):
        yp = self.w * x + self.b
        return yp