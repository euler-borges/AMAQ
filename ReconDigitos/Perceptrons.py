class Perceptron:
    def __init__(self, alvo):
        self.pesos_atuais = [0] * 49
        self.pesos_passados = [0] * 49
        self.bias = 0
        self.bias_passado = 0
        self.alpha = 1
        self.alvo = alvo
    
    #representa uma iteração do passo 2
    def treinar_perceptron(self, digito):
        saida_teorica = 1 if digito == self.alvo else -1
        #saida não bipolar
        saida_atual_st = self.bias + sum([digito[i] * self.pesos_atuais[i] for i in range(len(digito))])
  
        #saida bipolar
        saida_atual_ct = 1 if saida_atual_st >= 0 else -1

        if saida_atual_ct != saida_teorica:
            #atualizando bias
            self.bias += self.alpha * saida_teorica

            #atualizando pesos
            for i in range(len(digito)):
                self.pesos_atuais[i] += self.alpha * digito[i] * saida_teorica


    def atualizar_pesos_passados(self):
        self.pesos_passados = self.pesos_atuais
        self.bias_passado = self.bias


    def teste_de_igualdade(self):
        return self.pesos_atuais == self.pesos_passados and self.bias == self.bias_passado