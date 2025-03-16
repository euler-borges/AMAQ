import pandas
import matplotlib.pyplot as plt
import numpy as np
from neuronios import Neuronio_saida as Neuronio_saida, Neuronio_oculto as Neuronio_oculto


# Carregar os dados
file_path = "dados.csv"  

df = pandas.read_csv(file_path)
X = df.iloc[:, 0].values
y = df.iloc[:, -1].values
ciclos_t = 10000

camada_oculta = []
camada_saida = []

n_ocultos = 7
n_saida = 1

def sigmoide_bipolar(x):
    return (2 / (1 + np.exp(-x))) - 1
    
def derivada_sigmoide_bipolar(x):
    return 0.5 * (1 + sigmoide_bipolar(x)) * (1 - sigmoide_bipolar(x))


def criar_neuronios(n_ocultos, n_saida):
    #4 neuronios ocultos?
    for i in range(n_ocultos):
        camada_oculta.append(Neuronio_oculto())
        camada_oculta[i].ativacao(X[0])

    #1 neuronio de saida?
    for i in range(n_saida):
        camada_saida.append(Neuronio_saida(n_ocultos=n_ocultos))
        camada_saida[i].ativacao(camada_oculta)

def treino(ciclos_t):
    ciclos = 0
    while ciclos < ciclos_t:
        ciclos += 1
        for j in range(len(X)):
        #armazenando os resultados ocultos
            somas_ocultas = []
            resultados_ocultos = []
            for i in range(n_ocultos):
                somas_ocultas.append(camada_oculta[i].teste(X[j]))
                resultados_ocultos.append(sigmoide_bipolar(somas_ocultas[i]))

            # print(somas_ocultas)
            # print(resultados_ocultos)
            #calculando saida
            somas_saida = []
            resultados_saida = []
            for i in range(n_saida):
                # print(camada_saida[i].teste(resultados_ocultos))
                somas_saida.append(camada_saida[i].teste(resultados_ocultos))
                resultados_saida.append(sigmoide_bipolar(somas_saida[i]))

            #calculando erro
            

            for i in range(n_saida):
                #print(y[j], resultados_saida[i])
                erro = y[j] - resultados_saida[i]
                # print(erro)
                camada_saida[i].delta = erro * derivada_sigmoide_bipolar(somas_saida[i])
                for k in range(len(camada_saida[i].pesos)):    
                    camada_saida[i].termo_correcao_peso[k] = camada_saida[i].learning_rate * camada_saida[i].delta * resultados_ocultos[k]
                camada_saida[i].termo_correcao_bias = camada_saida[i].learning_rate * camada_saida[i].delta

            delta_calculado = camada_saida[0].delta
            #calcular erro oculto
            for i in range(n_ocultos):
                erro_oculto = delta_calculado * camada_saida[0].pesos[i]
                camada_oculta[i].delta = erro_oculto * derivada_sigmoide_bipolar(somas_ocultas[i])
                for k in range(len(camada_oculta[i].pesos)):
                    # print(camada_oculta[i].learning_rate, camada_oculta[i].delta, X[j])
                    camada_oculta[i].termo_correcao_peso[k] = camada_oculta[i].learning_rate * camada_oculta[i].delta * X[j]
                camada_oculta[i].termo_correcao_bias = camada_oculta[i].learning_rate * camada_oculta[i].delta

            #atualizar pesos
            for i in range(n_saida):
                for k in range(len(camada_saida[i].pesos)):
                    # print(i, k)
                    # print(camada_saida[i].pesos[k])
                    # print(camada_saida[i].termo_correcao_peso[k])
                    camada_saida[i].pesos[k] += camada_saida[i].termo_correcao_peso[k]
                camada_saida[i].bias += camada_saida[i].termo_correcao_bias

            for i in range(n_ocultos):
                for k in range(len(camada_oculta[i].pesos)):
                    camada_oculta[i].pesos[k] += camada_oculta[i].termo_correcao_peso[k]
                camada_oculta[i].bias += camada_oculta[i].termo_correcao_bias

def teste():
    for j in range(len(X)):
        #armazenando os resultados ocultos
        somas_ocultas = []
        resultados_ocultos = []
        for i in range(n_ocultos):
            somas_ocultas.append(camada_oculta[i].teste(X[j]))
            resultados_ocultos.append(sigmoide_bipolar(somas_ocultas[i]))

        #calculando saida
        somas_saida = []
        resultados_saida = []
        for i in range(n_saida):
            somas_saida.append(camada_saida[i].teste(resultados_ocultos))
            resultados_saida.append(sigmoide_bipolar(somas_saida[i]))

        # print("Entrada: ", X[j])
        print("Saida esperada: ", y[j])
        # print("Somas saida: ", somas_saida)
        print("Saida: ", resultados_saida)
        print("")

if __name__ == "__main__":
    criar_neuronios(n_ocultos, n_saida)
    treino(ciclos_t)
    teste()



