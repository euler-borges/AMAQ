# Nome do arquivo com as tabelas da verdade
arquivo_tabelas = "tabelas.txt"

# Inicializando pesos e bias
pesos = [0, 0]  # Para duas entradas
bias = 0
tabelas_corretas = []
tabelas_incorretas = []
tabelas_correta = True

# Função para aplicar a regra de Hebb
def aplicar_regra_hebb(entradas, saida):
    global pesos, bias
    for i in range(len(pesos)):
        pesos[i] += entradas[i] * saida  # Atualiza o peso com base na entrada e na saída
    bias += saida  # Atualiza o bias com a saída

# Processando o arquivo
with open(arquivo_tabelas, "r") as arquivo:
    #le as linhas do arquivo
    linhas = arquivo.readlines()
    #indice
    linha_atual = 0
    for linha in linhas:
        #atualiza o indice
        linha_atual += 1
        if linha_atual % 5 != 0:
            dados = linha.strip().split()
            #transformando as entradas em bipolares
            for i in range(len(dados)):
                if dados[i] == "0":
                    dados[i] = "-1"

            entradas = list(map(int, dados[:2]))  # As duas primeiras colunas são entradas

            saida = int(dados[2])  # A última coluna é a saída
            aplicar_regra_hebb(entradas, saida)
        else:
            # Exibindo os pesos e bias calculados
            tabela = linha_atual / 5
            print("Pesos finais para a tabela ", tabela, ": ", pesos)
            print("Bias final para a tabela ", tabela, ": ", bias)

            # Testando a rede com as entradas da tabela
            print("Testando a rede com as entradas da tabela ", tabela)

            for linha in linhas[linha_atual - 5:linha_atual - 1]:
                dados = linha.strip().split()
                for i in range(len(dados)):
                    if dados[i] == "0":
                        dados[i] = "-1"
                entradas = list(map(int, dados[:2]))
                saida = int(dados[2])

                # Calculando a saída da rede
                soma = sum([entradas[i] * pesos[i] for i in range(len(pesos))]) + bias
                if soma > 0:
                    saida_rede = 1
                else:
                    saida_rede = -1

                print("Entradas: ", entradas, " Saída esperada: ", saida, " Saída da rede: ", saida_rede)
                if saida == saida_rede:
                    continue
                else:
                    tabelas_correta = False
            
            if tabelas_correta:
                tabelas_corretas.append(tabela)
                print("Sucesso para a tabela ", tabela)
            else:
                tabelas_incorretas.append(tabela)
                print("Falha para a tabela ", tabela)

            # Reinicializando pesos e bias para a proxima tabela
            pesos = [0, 0]
            bias = 0
            tabelas_correta = True
            print()

print("Tabelas corretas: ", tabelas_corretas)
print("Tabelas incorretas: ", tabelas_incorretas)