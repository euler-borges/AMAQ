#nomes dos arquivos com as letras
T = "T.txt"
X = "X.txt"

# Inicializando pesos, bias e alpha
pesos = [0, 0, 0, 0, 0]  # Para conco entradas
bias = 0
alpha = 1

#basicamente a mesma coisa que a regra de hebb 
def aplicar_perceptron(entradas, saida):
    global pesos, bias, alpha
    while True:
        novos_pesos = [0, 0, 0, 0, 0]
        for i in range(len(pesos)):
            novos_pesos[i] += alpha * entradas[i] * saida  # Atualiza o peso com base na entrada e na saída
        novo_bias += alpha * saida  # Atualiza o bias com a saída
        if pesos == novos_pesos and novo_bias == bias:
            break


with open(T, "r") as arquivo:
    linhas = arquivo.readlines()
    for linha in linhas:
        dados = linha.strip().split()
        for i in range(len(dados)):
            if dados[i] == "0":
                dados[i] = "-1"

        entradas = list(map(int, dados[:5]))  # As cinco primeiras colunas são entradas

        saida = int(dados[5])  # A última coluna é a saída
        aplicar_perceptron(entradas, saida)