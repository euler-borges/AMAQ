#nomes dos arquivos com as letras
T = "T.txt"
X = "X.txt"

# Inicializando pesos, bias e alpha
pesos = [0] * 25 # Para cinco entradas por linha
bias = 0
alpha = 1
changed = True


#entrada sendo linhas 
def aplicar_regra_hebb_modificada(entradas, saida):
    global pesos, bias, alpha, changed
    #STEP4-------------------------------------------------------------------------------------------
    y = bias + sum([entradas[i] * pesos[i] for i in range(len(pesos))])
    y = 1 if y > 0 else -1
    #-----------------------------------------------------------------------------------------------
    #STEP5-------------------------------------------------------------------------------------------
    if y == saida:
        changed = False
    else:
        for i in range(len(pesos)):
            pesos[i] += pesos[i] + alpha * entradas[i] * saida  # Atualiza o peso com base na entrada e na saída
        bias += bias + alpha * saida  # Atualiza o bias com a saída
        changed = True
    #-----------------------------------------------------------------------------------------------


def trata_arquivo(arquivo, saida):
#STEP 3-------------------------------------------------------------------------------------------
    entradas = []
    linhas = arquivo.readlines()
    

    for linha in linhas:
        dados = linha.strip().split()

        # transformando as entradas em bipolares
        for i in range(len(dados)):
            if dados[i] == "0":
                dados[i] = "-1"
    
            entradas.append(int(dados[i]))  # As cinco primeiras colunas são entradas

#--------------------------------------------------------------------------------------------
    aplicar_regra_hebb_modificada(entradas, saida)

#basicamente a mesma coisa que a regra de hebb 
#entrada sendo o arquivo
def treinar_perceptron(arquivo_desejado, arquivo_indesejado):
    global pesos, bias, alpha
    dados = []
    novos_pesos = []
    novo_bias = 0

    try:
        while True:
            with open(arquivo_desejado, "r") as arquivo:
                trata_arquivo(arquivo, 1)


            with open(arquivo_indesejado, "r") as arquivo:
                trata_arquivo(arquivo, -1)
                
            if not changed:
                break

    except Exception as error:
        print("Erro ao processar o arquivo: ", error)


#função para decidir qual arquivo será utilizado
def escolher_arquivo():
    while True:
        try:            
            entrada = input("Qual arquivo deseja processar? (T ou X): ")
            if entrada in ["T", "X"]:
                entrada += ".txt"
                break
            else:
                print("Entrada inválida. Tente novamente.")
        except Exception as error:
            print("Erro ao processar a entrada: ", error)

    return entrada

def testar_rede(arquivo):
    global pesos, bias
    entradas = []
    try:
        with open(arquivo, "r") as arquivo:
            linhas = arquivo.readlines()
                
            while True:
                for linha in linhas:
                    dados = linha.strip().split()
                    # transformando as entradas em bipolares
                    for i in range(len(dados)):
                        if dados[i] == "0":
                            dados[i] = "-1"    
                        entradas.append(int(dados[i])) 
                soma = sum([entradas[i] * pesos[i] for i in range(len(pesos))]) + bias

                if soma > 0:
                    return 1
                else:
                    return -1

    except Exception as error:
        print("Erro ao processar o arquivo: ", error)

if __name__ == "__main__":
    entrada = escolher_arquivo()
    #arco de treinamento da rede com a letra escolhida
    if entrada == "T.txt":
        treinar_perceptron(T, X)
    else:
        treinar_perceptron(X, T)

    print("Pesos finais: ", pesos)
    print("Bias final: ", bias)

    # Testando a rede com as entradas da tabela
    while True:
        try:
            entrada = input("Qual arquivo deseja testar? (T ou X): ")
            if entrada in ["T", "X"]:
                entrada += ".txt"
                print(testar_rede(entrada))
            else:
                print("Entrada inválida. Tente novamente.")
        except Exception as error:
            print("Erro ao processar a entrada: ", error)