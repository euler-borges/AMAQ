import random
import numpy as np

class Neuronio:
    def __init__(self, learning_rate=0.001, epochs=100, tolerance=1e-3):
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.errors = []
        self.tolerance = tolerance
        self.pesos = None
        self.termo_correcao_peso = [0,0,0,0,0]
        self.termo_correcao_bias = 0
        self.delta = 0

    def ativacao(self, entradas):
        self.pesos = [random.random]
        self.pesos = self.pesos * 0.3

    def teste(self, entradas):
        soma = np.dot(self.pesos, entradas) + self.bias
        return soma

class Neuronio_saida(Neuronio):
    def __init__(self, epochs, learning_rate, tolerance):
        super().__init__(epochs, learning_rate, tolerance)

    def ativacao(self, entradas):
        self.pesos = [random.random() for i in range(4)]

    # def teste(self, entradas):
    #     print(entradas)
    #     print(self.pesos)
    #     soma = np.dot(self.pesos, entradas) + self.bias
    #     return soma



class Neuronio_oculto(Neuronio):
    def __init__(self, epochs, learning_rate, tolerance):
        super().__init__(epochs, learning_rate, tolerance)

    def ativacao(self, entradas):
        self.pesos = [random.random() for i in range(1)]