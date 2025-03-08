import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    def __init__(self):
        self.weights = None  # Make sure weights are initialized
    
    def least_squares_fit(self, X, Y):

        # Check Whether X and Y are Numpy Arrays and convert them to Numpy Arrays if not
        if type(X) != np.ndarray:
            X = np.array(X)
        if type(Y) != np.ndarray:
            Y = np.array(Y)
        
        X_extended = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias column
        # Check if X and Y have the same number of rows
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")
        
        self.weights = np.linalg.inv(np.transpose(X_extended) @ X_extended) @ np.transpose(X_extended) @ Y

    def gradient_descent_fit(self, X, Y):
        return None
    
    def predict(self, X):
        # Extend X to account for the bias term
        X_extended = np.c_[np.ones((X.shape[0], 1)), X]
        Y = X_extended @ self.weights 
        return Y
    
    def plot(self, X, Y):
        plt.scatter(X, Y)
        plt.plot(X, self.predict(X), color='red')
        plt.show()
    
# y = mx + c

test = LinearRegression()

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = np.array([2, 4, 6, 9, 3, 5, 7, 8, 10, 11])

new_X = np.array([6, 7, 8, 9, 10])
test.least_squares_fit(X, Y)
test.plot(X, Y)

print(test.predict(new_X))
