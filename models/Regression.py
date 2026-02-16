import pandas as pd
import numpy as np

class LogisticRegression:
    def __init__(self, x, y):
        self.bias = 0.0
        self.weight = 0.0
        self.x = self.x = np.array(x).reshape(-1, 1)
        self.y = self.y = np.array(y).reshape(-1, 1)
        
    def fit(self, epoch, lr):
        m = len(self.x)
        self.P = lambda za: 1 / (1 + np.exp(-za))
        self.x = self.x / 100
        for i in range(epoch):
            z = self.weight * self.x + self.bias
            yPred = self.P(z)
            
            error = yPred - self.y
            notDoneYet = (1 / m) * np.mean(error) * self.x
            tsTuff = (1 / m) * np.mean(error)

            self.weight -= lr * notDoneYet
            self.bias -= lr * tsTuff
            
        return self.weight, self.bias
    
    def predict(self, x):
        P = self.P(za=self.weight * x + self.bias) 
        return P
        
class LinearRegression:
    def __init__(self, x, y):
        self.bias = 0
        self.weight = 0
        self.x = self.x = np.array(x).reshape(-1, 1)
        self.y = self.y = np.array(y).reshape(-1, 1)
        
    def fit(self, epoch, lr):
        self.P = lambda w, b, x: w * x + b
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

def split_data(x, y, test_size, random_state):
    n = int(len(x) * test_size)
    x_test = x.sample(frac=test_size, random_state=random_state)
    y_test = y.sample(frac=test_size, random_state=random_state)
    x_train = x.drop(x_test.index)
    y_train = y.drop(y_test.index)

    return x_train, y_train, x_test, y_test


        

