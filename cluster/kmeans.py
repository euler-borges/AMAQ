import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from classes.ponto import Centroide, Ponto

'''
1- Importar os dados
2- Gerar os centroides aleatoriamente
3- Calcular a distância entre os pontos e os centroides
4- Atribuir os pontos ao centroide mais próximo
5- Atualizar os centroides com a média dos pontos atribuídos a eles
6- Repetir os passos 3 a 5 até que os centroides não mudem mais ou até atingir um número máximo de iterações
7- Plotar os pontos e os centroides finais
'''

####CONSTANTES####
max_iter = 1
n_centroides = 4

dados = pd.read_csv("observacoescluster.csv", sep=",", header=None)

X = dados.iloc[:, 0].values
Y = dados.iloc[:, 1].values


pontos = []
for i in range(len(X)):
    pontos.append(Ponto(X[i], Y[i]))

centroides = []
for i in range(n_centroides):
    centroides.append(Centroide(X, Y))

# Gerar os centroides aleatoriamente
check_iter = True
grupos = None

while check_iter:
    check_iter = False
    # Calcular a distância entre os pontos e os centroides
    for i in range(len(pontos)):
        for j in range(len(centroides)):
            pontos[i].distancias.append(pontos[i].distancia(centroides[j]))

    # Atribuir os pontos ao centroide mais próximo
    grupos = np.argmin([ponto.distancias for ponto in pontos], axis=1)
    
    # print(grupos)
    
    # Atualizar os centroides com a média dos pontos atribuídos a eles
    for i in range(n_centroides):
        erro_parcial = 0
        indices = np.where(grupos == i)[0]
        # print(indices)
    
        if len(indices) > 0:
            centroides[i].x_passado = centroides[i].x
            centroides[i].y_passado = centroides[i].y

            centroides[i].x = np.mean([pontos[idx].x for idx in indices])
            centroides[i].y = np.mean([pontos[idx].y for idx in indices])
            
            centroides[i].erro.append(sum([(pontos[idx].distancias[i])**2 for idx in indices]))

        check_iter = check_iter or (abs(centroides[i].x - centroides[i].x_passado) > 0.01) or (abs(centroides[i].y - centroides[i].y_passado) > 0.01)
    
    # Limpar as distâncias para a próxima iteração
    for ponto in pontos:
        ponto.distancias.clear()

# Plotar os pontos e os centroides finais

# Plotar os pontos e os centroides finais
cores = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'yellow', 'brown']  # Pode aumentar se tiver mais centroides

plt.figure(figsize=(12, 5))

# --- Subplot 1: clusters ---
plt.subplot(1, 2, 1)
for i in range(n_centroides):
    indices = np.where(grupos == i)[0]
    cluster_x = [pontos[idx].x for idx in indices]
    cluster_y = [pontos[idx].y for idx in indices]
    plt.scatter(cluster_x, cluster_y, c=cores[i], label=f'Cluster {i}')
    
    # Plotar o centroide
    plt.scatter(centroides[i].x, centroides[i].y, c='black', marker='X', s=100, edgecolors='white')

plt.title("Clusters Finais e Centroides")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# --- Subplot 2: evolução do erro ---
plt.subplot(1, 2, 2)
for i in range(n_centroides):
    plt.plot(centroides[i].erro, label=f'Erro Centroide {i}', color=cores[i])

plt.title("Evolução do Erro por Centroide")
plt.xlabel("Iteração")
plt.ylabel("Erro (Soma das Distâncias ao Quadrado)")
plt.legend()

plt.tight_layout()
plt.show()
