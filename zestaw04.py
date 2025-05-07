from MyGraph import *
from DrawGraph import *
from DirectedMyGraph import *
import random

# Przykładowe użycie:
if __name__ == "__main__":
    # Zadanie 1
    # Napisać program do kodowania grafów skierowanych (digrafów) i do generowania losowych digrafów z zespołu G(np).
    n = 7; p = 0.4
    g_list = generate_random_directed_graph_by_probability(n, p)
    print("Lista sąsiedztwa:")
    print(g_list)
    print()
    g_mat = DirectedAdjacencyMatrix(g_list)
    print("Macierz sąsiedztwa:")
    print(g_mat)
    print()
    g_inc = DirectedIncidenceMatrix(g_list)
    print("Macierz incydencji:")
    print(g_inc)


    import networkx as nx
    import matplotlib.pyplot as plt

    # konwersja do NetworkX DiGraph
    G = nx.DiGraph()
    for u in range(g_list.n):
        for v in g_list.adj[u]:
            G.add_edge(u, v)

    # ustawienie pozycji na okręgu
    pos = nx.circular_layout(G)

    # rysowanie
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, with_labels=True, arrows=True, node_color="lightblue", edge_color="gray", 
            font_weight="bold", font_size=10)
    plt.title(f"Digraf G({n}, {p})")
    plt.axis("off")
    plt.show()
