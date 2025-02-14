class Perceptron:
    def __init__(self, alvo):
        self.pesos = [0] * 49
        self.bias = 0
        self.alpha = 1
        self.alvo = alvo
