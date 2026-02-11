import math 

class fit:
    def __init__(self, b0, b1, data, lr, yt):
        self.b0 = b0
        self.b1 = b1
        self.data = data
        self.lr = lr
        self.yt = yt

    def gradientDescent(self):
        self.z = self.b0 + (self.b1 * self.data)
        yp = 1 / (1 + math.e ** - self.z)
        self.lossb1 = (yp - self.yt) * self.data 
        self.lossb0 = (yp - self.yt)

    def update(self):
        self.b1 = (self.z - self.lr) * self.lossb1
        self.b0 = (self.z - self.lr) * self.lossb0

        self.z = self.b0 + (self.b1 * self.data)
        yp = 1 / (1 + math.e ** - self.z)
        yp = float(yp)

        return yp
    
model = fit(b0=0, b1=0, data=1, lr=0.1, yt=1)
model.gradientDescent()
model = model.update()
print(model)