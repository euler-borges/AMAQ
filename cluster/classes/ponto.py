import random

class Ponto():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.distancias = []


    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        if isinstance(other, ponto):
            return self.x == other.x and self.y == other.y
        return False

    def distancia(self, outro_ponto):
        """
        Calcula a distância entre dois pontos no espaço 2D.
        
        :param outro_ponto: Outro ponto para o qual a distância será calculada.
        :return: Distância entre os dois pontos.
        """
        return ((self.x - outro_ponto.x) ** 2 + (self.y - outro_ponto.y) ** 2) ** 0.5
    











class Centroide(Ponto):
    """
    Classe que representa um centroide em um espaço de duas dimensões.
    """

    def __init__(self, X, Y):
        """
        Inicializa o centroide com as coordenadas x e y.

        :param x: Coordenada x do centroide.
        :param y: Coordenada y do centroide.
        """
        self.x = random.random() * (X.max() - X.min()) + X.min()
        self.y = random.random() * (Y.max() - Y.min()) + Y.min()
        self.x_passado = self.x
        self.y_passado = self.y
        self.distancias = []
        self.pontos = []
        self.erro = []
        
    def __str__(self):
        """
        Retorna uma representação em string do centroide.

        :return: String representando o centroide.
        """
        return f"Centroide({self.x}, {self.y})"
    
    def __eq__(self, other):
        return super().__eq__(other)