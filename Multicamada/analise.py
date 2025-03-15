import pandas
import matplotlib.pyplot as plt
import numpy as np
from neuronios import Neuronio_saida as Neuronio_saida, Neuronio_oculto as Neuronio_oculto


# Carregar os dados
file_path = "dados.csv"  

df = pandas.read_csv(file_path)
X = df.iloc[:, 0].values
y = df.iloc[:, -1].values
ciclos = 0

camada_oculta = []
camada_saida = []

n_ocultos = 4
n_saida = 1

def sigmoide_bipolar(x):
    return (2 / (1 + np.exp(-x))) - 1
    
def derivada_sigmoide_bipolar(x):
    return 0.5 * (1 + sigmoide_bipolar(x)) * (1 - sigmoide_bipolar(x))


def criar_neuronios(n_ocultos, n_saida):
    #4 neuronios ocultos?
    for i in range(n_ocultos):
        camada_oculta.append(Neuronio_oculto(epochs=1000, learning_rate=0.001, tolerance=1e-3))
        camada_oculta[i].ativacao(X[0])

    #1 neuronio de saida?
    for i in range(n_saida):
        camada_saida.append(Neuronio_saida(epochs=1000, learning_rate=0.001, tolerance=1e-3))
        camada_saida[i].ativacao(camada_oculta)

def treino(ciclos):
    while ciclos < 100:
        ciclos += 1
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

            #calculando erro
            

            for i in range(n_saida):
                erro = y[j] - resultados_saida[i]
                camada_saida[i].delta = erro * derivada_sigmoide_bipolar(somas_saida[i])
                for k in range(len(camada_saida[i].pesos)):    
                    camada_saida[i].termo_correcao_peso[k] = camada_saida[i].learning_rate * camada_saida[i].delta * resultados_ocultos[i]
                camada_saida[i].termo_correcao_bias = camada_saida[i].learning_rate * camada_saida[i].delta

            delta_calculado = camada_saida[0].delta
            #calcular erro oculto
            for i in range(n_ocultos):
                erro_oculto = delta_calculado * camada_saida[0].pesos[i]
                camada_oculta[i].delta = erro_oculto * derivada_sigmoide_bipolar(somas_ocultas[i])
                for k in range(len(camada_oculta[i].pesos)):
                    camada_oculta[i].termo_correcao_peso[k] = camada_oculta[i].learning_rate * camada_oculta[i].delta * X[j]
                camada_oculta[i].termo_correcao_bias = camada_oculta[i].learning_rate * camada_oculta[i].delta

            #atualizar pesos
            for i in range(n_saida):
                for k in range(len(camada_saida[i].pesos)):
                    print(i, k)
                    print(camada_saida[i].pesos[k])
                    print(camada_saida[i].termo_correcao_peso[k])
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

        print("Saida: ", resultados_saida)


if __name__ == "__main__":
    criar_neuronios(n_ocultos, n_saida)
    treino(ciclos)
    teste()





# meanX = X.mean()
# meanY = y.mean()
# # Criar e treinar o modelo
# adaline = Adaline(learning_rate=0.001, epochs=10000, tolerance=1e-1)
# adaline.fit(X, y)

# #
# n_samples, n_features = X.shape

# # Calculando ccoeficientes da regressão linear
# x = X[:, 0]
# b = (n_samples * (x.dot(y)) - x.sum() * y.sum()) / (n_samples * (x.dot(x)) - x.sum() ** 2)
# a = y.mean() - b * x.mean()


# # # Plotar erro quadrático total
# # plt.plot(range(1, len(adaline.errors) + 1), adaline.errors, marker='o')
# # plt.xlabel('Épocas')
# # plt.ylabel('Erro Quadrático Médio')
# # plt.title('Evolução do Erro Durante o Treinamento')
# # plt.show()


# # Testar a rede treinada
# predictions = adaline.predict(X)
# # print("Saídas previstas:", predictions)



# fig, axs = plt.subplots(2,2)
# # Plotar a linha adaline
# axs[0, 0].plot(X, predictions, marker='.')
# axs[0, 0].set_xlabel('X')
# axs[0, 0].set_ylabel('Y')
# axs[0, 0].set_title('Saídas Previstas pela Adaline')

# # Fornecida
# axs[1, 1].plot(X, y, marker='1')
# axs[1, 1].set_xlabel('X')
# axs[1, 1].set_ylabel('Y')
# axs[1, 1].set_title('Saídas Fornecidas')

# # Plotar a linha de regressão linear da fórmula
# y_formula = b * X + a
# axs[1, 0].plot(X, y_formula, marker='2')
# axs[1, 0].set_xlabel('X')
# axs[1, 0].set_ylabel('Y')
# axs[1, 0].set_title('Regressão Linear da Fórmula')

# pierson = (n_samples * (x.dot(y)) - x.sum() * y.sum()) / (((n_samples * (x.dot(x)) - x.sum() ** 2) * (n_samples * (y.dot(y)) - y.sum() ** 2)) ** 0.5)
# determinacao = pierson ** 2

# print("Coeficiente de Pierson: ", pierson)
# print("Coeficiente de determinação: ", determinacao)
# # plt.title('Evolução do Erro Durante o Treinamento')
# plt.show()