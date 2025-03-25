import pandas
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from neuronios import Neuronio_saida as Neuronio_saida, Neuronio_oculto as Neuronio_oculto, lista as lista

N_CICLOS = 1
lista_inicializacao = [10, 7, 10]
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar os pixels para o intervalo [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Achatar as imagens (de 28x28 para 784)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# def sigmoide_bipolar(x):
#     return (2 / (1 + np.exp(-x))) - 1
    
# def derivada_sigmoide_bipolar(x):
#     return 0.5 * (1 + sigmoide_bipolar(x)) * (1 - sigmoide_bipolar(x))

def inicializacao(lista_inicializacao):

    #definindo a lista de inicialização
    lista_neuronios = lista(lista_inicializacao)
    lista_neuronios.inicializacao()
    #neuronios com pesos sendo gerados adequadamente
    return lista_neuronios.lista

def treino(ciclos_t, lista_neuronios):
    ciclos = 0
    while ciclos < ciclos_t:
        ciclos += 1
        #para cada training pair
        for i in range(len(x_train)):
            #tratando das camadas
            for j in range(len(lista_neuronios)):
                #j representa a camada
                #primeira camada
                if j == 0:
                    for neuronio in lista_neuronios[j]:
                        neuronio.calcula_soma_nao_ativada(x_train[i])
                        neuronio.calcula_soma_ativada()
                #ultima camada
                elif j == len(lista_neuronios) - 1:
                    entradas = []
                    for neuronio_anterior in lista_neuronios[j-1]:
                        entradas.append(neuronio_anterior.soma_ativa)
                    for neuronio in lista_neuronios[j]:
                        neuronio.calcula_soma_nao_ativada(entradas)
                        neuronio.calcula_soma_ativada()
                #demais camadas
                else:
                    entradas = []
                    for neuronio_anterior in lista_neuronios[j-1]:
                        entradas.append(neuronio_anterior.soma_ativa)
                    for neuronio in lista_neuronios[j]:
                        neuronio.calcula_soma_nao_ativada(entradas)
                        neuronio.calcula_soma_ativada()
            #retropropagação do erro
            

if __name__ == "__main__":
    lista_neuronios =  inicializacao(lista_inicializacao)
    treino(N_CICLOS, lista_neuronios)









