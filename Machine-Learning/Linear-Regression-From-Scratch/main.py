import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def least_squares_fit(self, X, Y):
        X, Y = self.convert_to_numpy(X, Y)
        
        X_extended = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias column
        # Check if X and Y have the same number of rows
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")
        
        self.weights = np.linalg.inv(np.transpose(X_extended) @ X_extended) @ np.transpose(X_extended) @ Y
        self.bias = self.weights[0]  # First element is the bias term
        self.weights = self.weights[1:]  # Remaining elements are the weights

    def gradient_descent_fit(self, X, Y, learning_rate=0.01, max_iterations=1000):
        X, Y = self.convert_to_numpy(X, Y)
        
        # Ensure X and Y are 2D arrays (n_samples, n_features) and (n_samples, n_outputs) respectively
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Make sure X is a 2D array (n_samples, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)  # Make sure Y is a 2D array (n_samples, 1)

        # Initialize weights with zeros
        self.weights = np.zeros((X.shape[1], Y.shape[1]))  # (features, outputs)
        self.bias = np.zeros((1, Y.shape[1]))  # (1, outputs)

        # Gradient Descent Algorithm 
        for i in range(max_iterations):
            # Forward pass
            Y_hat = X @ self.weights + self.bias
            
            # Backward pass
            dW = 2 / X.shape[0] * (X.T @ (Y_hat - Y))
            dB = 2 / X.shape[0] * np.sum(Y_hat - Y, axis=0, keepdims=True)

            # Update weights and bias
            self.weights -= learning_rate * dW
            self.bias -= learning_rate * dB

    def cost_function(self, Y, Y_hat):
        # MSE cost function
        return 1 / Y.shape[0] * np.sum((Y - Y_hat) ** 2)

    def convert_to_numpy(self, X, Y=None):
        if type(X) != np.ndarray:
            X = np.array(X)
        if type(Y) != np.ndarray and Y is not None:
            Y = np.array(Y)
        return X, Y
    
    def predict(self, X):
        X, _ = self.convert_to_numpy(X)
        
        # Ensure X is 2D (n_samples, 1) if it's 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.weights is not None and self.bias is not None:
            Y_hat = X @ self.weights + self.bias
        else:
            X_extended = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias column
            Y_hat = X_extended @ self.weights  # For least squares prediction without the bias term

        return Y_hat
    
    def plot(self, X, Y):
        plt.scatter(X, Y)
        plt.plot(X, self.predict(X), color='red')
        plt.show()

test = LinearRegression()

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
Y = np.array([[2], [4], [6], [9], [3], [5], [7], [8], [10], [11]])

new_X = np.array([[6], [7], [8], [9], [10]])
test.gradient_descent_fit(X, Y, learning_rate=0.01, max_iterations=100000)
test.plot(X, Y)

print(test.predict(new_X))

test.least_squares_fit(X, Y)
test.plot(X, Y)
print(test.predict(new_X))
