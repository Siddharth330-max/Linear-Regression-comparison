import pandas as pd  
import numpy as np  

class LinearRegression:
    def __init__(self,lr = 0.01 , n_iter = 1000):
        self.lr = lr                 ##lr -> Learning Rate
        self.n_iter =n_iter          ##n_iter -> number of iterations or number of times the algorithm will repeat
        self.weights = None  
        self.bias = None  

    def fit(self ,X,y):
        n_samples , n_features = X.shape  

        self.weights = np.zeros(n_features)    ##total weights must be equal to number of features
                                                ##Both 'weights' and 'bias' must be initialised with 0  
        self.bias = 0                           ##initalisation happens for once only

        for _ in range(self.n_iter):
            ##We have to rn this algorithm for n_iter times to find the best 'weights' and 'bias'

            ##Step -1  -> FInding y_pred
            y_pred = np.dot(X , self.weights) + self.bias  

            ##Step -2 -> Gradient Descent  
            dw =(1/n_samples) * np.dot(X.T , (y_pred -y)) 
            db = (1/n_samples) * np.sum(y_pred - y)     

            ##Step -3 -> Updating weights and bias
            self.weights = self.weights - self.lr * dw  
            self.bias = self.bias - self.lr * db    

    def predict(self , X):
        y_pred = np.dot(X ,self.weights) + self.bias 
        return y_pred



        