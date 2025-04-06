import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Carregar o CSV
df = pd.read_csv("iris_data.csv", sep=";", header=None)

# Separar entradas (X) e saída (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values  # Mantém como vetor de rótulos

# Normalizar entradas
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Juntar entradas normalizadas com a saída original (classe)
df_final = pd.DataFrame(X_normalized, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df_final['class'] = y

# Salvar em novo CSV
df_final.to_csv("iris_preparado.csv", index=False)

print("✅ Arquivo 'iris_preparado.csv' criado com sucesso!")
