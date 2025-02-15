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
        PERCEPTRONS.append(Perceptron(digits_bitmap[digit]))

def testar_rede():
    pass

if __name__ == "__main__":
    main()

