import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=50, tolerance=1e-3):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.errors = []
        self.tolerance = tolerance
        self.past_weights = None
    
    def activation(self, x):
        return np.where(x >= 0, 1, -1)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.past_weights = [1000] * n_features
        self.bias = 0
        
        while True:
            
            net_input = np.dot(X, self.weights) + self.bias
            output = net_input
            errors = y - output
            
            
            self.weights += self.learning_rate * X.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()
            
            mse = (errors**2).mean()
            self.errors.append(mse)

            for weight, past_weight in zip(self.weights, self.past_weights):

                if abs(weight - past_weight) < self.tolerance:
                    return

    def predict(self, X):
        net_input = np.dot(X, self.weights) + self.bias
        return self.activation(net_input)