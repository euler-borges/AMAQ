import numpy as np

class Neuronio:
    def __init__(self, n_entradas):
        self.pesos = np.random.randn(n_entradas)
        self.bias = np.random.randn()
    
    def sigmoide_bipolar(self, x):
        return (2 / (1 + np.exp(-x))) - 1
    
    def ativacao(self, entradas):
        soma = np.dot(self.pesos, entradas) + self.bias
        return self.sigmoide_bipolar(soma)

class Neuronio_saida(Neuronio):
    def __init__(self, n_entradas):
        super().__init__(n_entradas)

class Neuronio_oculto(Neuronio):
    def __init__(self, n_entradas):
        super().__init__(n_entradas)
