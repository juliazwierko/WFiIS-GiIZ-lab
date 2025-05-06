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
    print('*'*100)
    
    # Zadanie 2
    # Zaimplementować algorytm Dijkstry do znajdowanian ajkrótszych ścieżek 
    # od zadanego wierzchołka do pozostałych wierzchołków i zastosować
    # go do grafu z zadania pierwszego, w którym wagi krawędzi interpreto-
    # wane są jako odległości wierzchołków. Wypisać wszystkie najkrótsze
    # ścieżki od danego wierzchołka i ich długości.
    distances, paths = dijkstra(g, 0, print_paths=True)
    print('*'*100)
    
    # Zadanie 3
    # Wyznaczyć macierz odległości miedzy wszystkimi parami wierzchołków na tym grafie.
    M = get_distance_matrix(g)
    print("\nDistance matrix:")
    print_matrix_nicely(M)
    print('*'*100)
    
    # Zadanie 4
    # Wyznaczyć centrum grafu, to znaczy wierzchołek, którego suma odległości do pozostałych wierzchołków jest minimalna. 
    # Wyznaczyć centrum minimax, to znaczy wierzchołek, którego odległość do najdalszego wierzchołka jest minimalna.
    center, minimax = find_graph_center_and_minimax_based_on_distance_matrix(M)
    print(f'Centrum = {center[0]} (suma odleglosci: {center[1]})')
    print(f'Centrum minimax = {minimax[0]} (odleglosc od najdalszego: {minimax[1]})')
    print('*'*100)
    
    # Zadanie 5
    # Wyznaczyć minimalne drzewo rozpinające (algorytm Prima lub Kruskala).
    T, W, mst_prim = prim(g, 0)
    print('Prim:')
    print(f'{T=}')
    print(f'{W=}')
    print(f'{mst_prim=}')
    print('Kruskal:')
    mst_kruskal = kruskal(g)
    print(f'{mst_kruskal=}')
    Draw(g, "graph_mst_prim", "Random Connected Graph with MST Prim", output_dir="outputs/03", with_weights=True, mst= mst_prim)
    Draw(g, "graph_mst_kruskal", "Random Connected Graph with MST Kruskal", output_dir="outputs/03", with_weights=True, mst= mst_kruskal)