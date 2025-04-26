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
    
    # Zadanie 2
    # Zaimplementować algorytm Dijkstry do znajdowanian ajkrótszych ścieżek 
    # od zadanego wierzchołka do pozostałych wierzchołków i zastosować
    # go do grafu z zadania pierwszego, w którym wagi krawędzi interpreto-
    # wane są jako odległości wierzchołków. Wypisać wszystkie najkrótsze
    # ścieżki od danego wierzchołka i ich długości.
    distances, paths = dijkstra(g, 0)
    
    # Zadanie 3
    # Wyznaczyć macierz odległości miedzy wszystkimi parami wierzchołków na tym grafie.
    M = get_distance_matrix(g)
    print("\nDistance matrix:")
    print_matrix_nicely(M)
    
    # Zadanie 4
    # Wyznaczyć centrum grafu, to znaczy wierzchołek, którego suma od-
    # ległości do pozostałych wierzchołków jest minimalna. Wyznaczyć cen-
    # trumminimax, toznaczywierzchołek, któregoodległośćdonajdalszego
    # wierzchołka jest minimalna.
    
    