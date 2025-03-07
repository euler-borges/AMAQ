import pandas
import matplotlib.pyplot as plt
from adaline import Adaline as Adaline


# Carregar os dados
file_path = "bd.csv"  

df = pandas.read_csv(file_path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
meanX = X.mean()
meanY = y.mean()
# Criar e treinar o modelo
adaline = Adaline(learning_rate=0.001, epochs=10000, tolerance=1e-1)
adaline.fit(X, y)

#
n_samples, n_features = X.shape

# Calculando ccoeficientes da regressão linear
x = X[:, 0]
b = (n_samples * (x.dot(y)) - x.sum() * y.sum()) / (n_samples * (x.dot(x)) - x.sum() ** 2)
a = y.mean() - b * x.mean()


# # Plotar erro quadrático total
# plt.plot(range(1, len(adaline.errors) + 1), adaline.errors, marker='o')
# plt.xlabel('Épocas')
# plt.ylabel('Erro Quadrático Médio')
# plt.title('Evolução do Erro Durante o Treinamento')
# plt.show()


# Testar a rede treinada
predictions = adaline.predict(X)
# print("Saídas previstas:", predictions)



fig, axs = plt.subplots(2,2)
# Plotar a linha adaline
axs[0, 0].plot(X, predictions, marker='.')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].set_title('Saídas Previstas pela Adaline')

# Fornecida
axs[1, 1].plot(X, y, marker='1')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')
axs[1, 1].set_title('Saídas Fornecidas')

# Plotar a linha de regressão linear da fórmula
y_formula = b * X + a
axs[1, 0].plot(X, y_formula, marker='2')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')
axs[1, 0].set_title('Regressão Linear da Fórmula')

pierson = (n_samples * (x.dot(y)) - x.sum() * y.sum()) / (((n_samples * (x.dot(x)) - x.sum() ** 2) * (n_samples * (y.dot(y)) - y.sum() ** 2)) ** 0.5)
determinacao = pierson ** 2

print("Coeficiente de Pierson: ", pierson)
print("Coeficiente de determinação: ", determinacao)
# plt.title('Evolução do Erro Durante o Treinamento')
plt.show()