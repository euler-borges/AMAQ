import pandas
import matplotlib.pyplot as plt
from adaline import Adaline as Adaline


# Carregar os dados
file_path = "b2.csv"  

df = pandas.read_csv(file_path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# Criar e treinar o modelo
adaline = Adaline(learning_rate=0.01, epochs=50, tolerance=1e-3)
adaline.fit(X, y)

# Plotar erro quadrático total
plt.plot(range(1, len(adaline.errors) + 1), adaline.errors, marker='o')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio')
plt.title('Evolução do Erro Durante o Treinamento')
plt.show()

# Testar a rede treinada
predictions = adaline.predict(X)
print("Saídas previstas:", predictions)
