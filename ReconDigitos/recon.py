from digits_bitmap import digits_bitmap
from Perceptrons import Perceptron

PERCEPTRONS = []

def main():
    #inicia os perceptrons necessários
    cria_perceptrons()

    #treina os perceptrons
    for perceptron in PERCEPTRONS:
        while True:
            for digit in digits_bitmap:
                perceptron.treinar_perceptron(digits_bitmap[digit])
            if perceptron.teste_de_igualdade():
                break
            else:
                perceptron.atualizar_pesos_passados()


    testar_rede()




def cria_perceptrons():
    for digit in digits_bitmap:
        PERCEPTRONS.append(Perceptron(digits_bitmap[digit], digit))

def testar_rede():
    try:

        while True:
            entrada = input("Digite um dígito: ")

            if entrada in digits_bitmap:
                entrada_array = digits_bitmap[entrada]
                for perceptron in PERCEPTRONS:
                    saida = perceptron.testar_perceptron(entrada_array)
                    print(f"O perceptron {perceptron.string} classificou o dígito {entrada} como {saida} \n \n")
                    
            else:
                print("Dígito inválido. Tente novamente.")

    except KeyboardInterrupt:
        print("Programa encerrado.")
        exit(0)
    except Exception as error:
        print("Algo deu errado: ", error)

if __name__ == "__main__":
    main()

