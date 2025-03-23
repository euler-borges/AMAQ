import random
import numpy as np

#lista de listas de neuronio
#cada entrada possui a lista dos neuronios da camada x
class lista:
    def __init__(self, lista_inicializacao):
        self.lista_inicializacao = lista_inicializacao
        self.lista = []

    def inicializacao(self):
        #olha as entradas da lista de inicialização
        for j in range(len(self.lista_inicializacao)):
            entrada = self.lista_inicializacao[j]
            lista_temp = []
            #primeira entrada, neuronios de entrada
            if j == 0:
                for i in range(entrada):
                    lista_temp.append(Neuronio_entrada())
                    lista_temp[i].ativacao()
                self.lista.append(lista_temp)
            elif j != len(self.lista_inicializacao) - 1:
                for i in range(entrada):
                    lista_temp.append(Neuronio_oculto())
                    lista_temp[i].ativacao(self.lista_inicializacao[j-1])
                self.lista.append(lista_temp)
            else:
                for i in range(entrada):
                    lista_temp.append(Neuronio_saida())
                    lista_temp[i].ativacao(self.lista_inicializacao[j-1])
                self.lista.append(lista_temp)


class Neuronio:
    def __init__(self):
        self.bias = 0
        self.learning_rate = 0.02
        self.epochs = 1000
        self.errors = []
        self.tolerance = 1e-3
        self.pesos = None
        self.termo_correcao_peso = [0]* 10
        self.termo_correcao_bias = 0
        self.delta = 0

    # def teste(self, entradas):
    #     print(entradas)
    #     print(self.pesos)
    #     print(self.bias)
    #     soma = np.dot(self.pesos, entradas) + self.bias
    #     print(soma[0])
    #     return soma[0]

class Neuronio_entrada(Neuronio):
    def __init__(self):
        super().__init__()

    def ativacao(self):
        self.pesos = [(random.random() * 0.3) for _ in range(784)]
        self.bias = random.random() * 0.1

    def teste(self, entradas):
        soma = np.dot(self.pesos, entradas) + self.bias
        return soma



class Neuronio_saida(Neuronio):
    def __init__(self):
        super().__init__()

    def ativacao(self, n_entradas):
        self.pesos = [(random.random() * 0.3) for _ in range(n_entradas)]
        self.bias = random.random() * 0.1

    def teste(self, entradas):
        soma = np.dot(self.pesos, entradas) + self.bias
        return soma



class Neuronio_oculto(Neuronio):
    def __init__(self):
        super().__init__()

    def ativacao(self, n_entradas):
        self.pesos = [(random.random() * 0.3) for _ in range(n_entradas)]
        self.bias = random.random() * 0.1

    def teste(self, entradas):
        soma = np.dot(self.pesos, entradas) + self.bias
        return soma[0]
