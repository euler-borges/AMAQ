import pandas
import matplotlib.pyplot as plt
import numpy as np
from time import time
from neuronios import Neuronio_saida as Neuronio_saida, Neuronio_oculto as Neuronio_oculto, lista as lista
import pandas as pd

N_CICLOS = 1500
lista_inicializacao = [15, 5, 10]


# Carregar o CSV para um DataFrame
df = pd.read_csv("iris_preparado.csv")

X = df.iloc[:, :-1].values  # Todas as colunas, menos a última (entradas normalizadas)
y = df.iloc[:, -1].values   # Apenas a última coluna (classe)




def inicializacao(lista_inicializacao):

    #definindo a lista de inicialização
    lista_camadas_neuronios = lista(lista_inicializacao)
    lista_camadas_neuronios.inicializacao()
    #neuronios com pesos sendo gerados adequadamente
    return lista_camadas_neuronios.lista

def treino(ciclos_t, lista_camadas_neuronios):
    ciclos = 0
    while ciclos < ciclos_t:
        ciclos += 1
        #para cada training pair
        for i in range(len(X)):
            #tratando das camadas
            for j in range(len(lista_camadas_neuronios)):
                #j representa a camada
                #primeira camada
                if j == 0:
                    for neuronio in lista_camadas_neuronios[j]:
                        neuronio.calcula_soma_nao_ativada(X[i])
                        neuronio.calcula_soma_ativada()
                #ultima camada
                elif j == len(lista_camadas_neuronios) - 1:
                    entradas = []
                    for neuronio_anterior in lista_camadas_neuronios[j-1]:
                        entradas.append(neuronio_anterior.soma_ativa)
                    for neuronio in lista_camadas_neuronios[j]:
                        neuronio.calcula_soma_nao_ativada(entradas)
                        neuronio.calcula_soma_ativada()
                #demais camadas
                else:
                    entradas = []
                    for neuronio_anterior in lista_camadas_neuronios[j-1]:
                        entradas.append(neuronio_anterior.soma_ativa)
                    for neuronio in lista_camadas_neuronios[j]:
                        neuronio.calcula_soma_nao_ativada(entradas)
                        neuronio.calcula_soma_ativada()
            #retropropagação do erro
            for j in range(len(lista_camadas_neuronios)):
                #tratando os neuronios de saida
                indice = len(lista_camadas_neuronios) - 1 - j
                # print(j, indice)
                if j == 0:
                    for neuronio in lista_camadas_neuronios[indice]:
                        neuronio.calcula_delta(y[i])
                        neuronio.calcula_termo_correcao(lista_camadas_neuronios[indice-1])
                #neuronios ocultos
                elif j != len(lista_camadas_neuronios) - 1:
                    for k in range(len(lista_camadas_neuronios[indice])):
                        neuronio = lista_camadas_neuronios[indice][k]
                        neuronio.calcula_delta(lista_camadas_neuronios[indice+1], k)
                        neuronio.calcula_termo_correcao(lista_camadas_neuronios[indice-1])
                #neuronios de entrada
                else:
                    for k in range(len(lista_camadas_neuronios[indice])):
                        neuronio = lista_camadas_neuronios[indice][k]
                        neuronio.calcula_delta(lista_camadas_neuronios[indice+1], k)
                        neuronio.calcula_termo_correcao(X[i])


            #atualizando os pesos e bias
            for camada in lista_camadas_neuronios:
                for neuronio in camada:
                    neuronio.atualiza_pesos()
        print("Ciclo:", ciclos, "de", ciclos_t, ".", time(), "segundos.")
    print("Treinamento concluído.")

def teste_unitario(entrada, valor_esperado):
    for camada in lista_camadas_neuronios:
        proxima_camada = []
        for neuronio in camada:
            proxima_camada.append(neuronio.fornece_resultado(entrada))
        entrada = proxima_camada
    # depois de passar a entrada pela rede, analisar qual neuronio forneceu o maior resultado
    maior = -1
    indice_maior = -1
    for i in range(len(entrada)):
        if entrada[i] > maior:
            maior = entrada[i]
            indice_maior = i

    # verificar se o resultado é igual ao esperado
    if indice_maior == valor_esperado:
        return True
    else:
        return False

def teste_total():
    # Teste do modelo com os dados de teste
    acertos = 0
    print("Iniciando teste total(conjunto de treinamento)...")
    for i in range(len(X)):
        if teste_unitario(X[i], y[i]):
            acertos += 1
    print("Total de acertos:", acertos, "de", len(X), "imagens.")
    print("Acurácia:", acertos / len(X) * 100, "%")



if __name__ == "__main__":
    lista_camadas_neuronios =  inicializacao(lista_inicializacao)
    treino(N_CICLOS, lista_camadas_neuronios)
    teste_total()








