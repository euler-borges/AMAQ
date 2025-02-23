import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.001, epochs=100, tolerance=1e-3):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.errors = []
        self.tolerance = tolerance
    
    def activation(self, x):
        return np.where(x >= 0, 1, -1)

    def fit(self, X, Y):
       

        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01  # Inicializar pesos pequenos
        self.bias = np.random.randn() * 0.01  # Inicializar bias pequeno
        y = np.zeros(n_samples)  
        

        for epoch in range(self.epochs):
            for i in range(n_samples):
                # print(X[i])
                # print(self.weights)
                # print(np.dot(X[i], self.weights))
                y[i] = self.bias + np.dot(X[i], self.weights)  
                
                difference = Y[i] - y[i]
                self.bias += self.learning_rate * difference
                self.weights += self.learning_rate * difference * X[i]

            self.errors.append(np.mean((Y - y) ** 2))  
            
            if self.errors[-1] < self.tolerance:
                break  

    def predict(self, X):
        net_input = np.dot(X, self.weights) + self.bias
        return self.activation(net_input)
