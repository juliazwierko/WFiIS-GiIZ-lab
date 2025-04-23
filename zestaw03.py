from MyGraph import *
from DrawGraph import *
import random

if __name__ == '__main__':
    # Zadanie 1
    # Wygenerować spójny graf losowy.
    # Przypisać każdej krawędzi tego grafu losową wagę będącą liczbą naturalną z zakresu 1 do 10.
    g  = generate_random_connected_graph(8, 10)
    g.add_weights(lambda : random.randint(1, 10))
    Draw(g, "random_connected_graph", "Random Connected Graph", output_dir="outputs/03", with_weights=True)