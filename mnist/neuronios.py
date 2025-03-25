import random
import numpy as np


def sigmoide_bipolar(x):
    return (2 / (1 + np.exp(-x))) - 1
    
def derivada_sigmoide_bipolar(x):
    return 0.5 * (1 + sigmoide_bipolar(x)) * (1 - sigmoide_bipolar(x))


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
                    lista_temp.append(Neuronio_saida(i))
                    lista_temp[i].ativacao(self.lista_inicializacao[j-1])
                self.lista.append(lista_temp)


class Neuronio:
    def __init__(self):
        self.bias = 0
        self.learning_rate = 0.01
        self.epochs = 1000
        self.errors = []
        self.tolerance = 1e-3
        self.pesos = None
        self.termo_correcao_peso = [0]* 10
        self.termo_correcao_bias = 0
        self.delta = 0
        self.soma_nao_ativa = 0
        self.soma_ativa = 0

    def calcula_soma_nao_ativada(self, entradas):
        self.soma_nao_ativa = np.dot(self.pesos, entradas) + self.bias

    def calcula_soma_ativada(self):
        self.soma_ativa = sigmoide_bipolar(self.soma_nao_ativa)

class Neuronio_entrada(Neuronio):
    def __init__(self):
        super().__init__()

    def ativacao(self):
        self.pesos = [(random.random() * 0.3) for _ in range(784)]
        self.bias = random.random() * 0.1




class Neuronio_saida(Neuronio):
    def __init__(self, esperado):
        super().__init__()
        self.esperado = esperado

    def ativacao(self, n_entradas):
        self.pesos = [(random.random() * 0.3) for _ in range(n_entradas)]
        self.bias = random.random() * 0.1




class Neuronio_oculto(Neuronio):
    def __init__(self):
        super().__init__()

    def ativacao(self, n_entradas):
        self.pesos = [(random.random() * 0.3) for _ in range(n_entradas)]
        self.bias = random.random() * 0.1
