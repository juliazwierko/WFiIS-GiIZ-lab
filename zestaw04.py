from MyGraph import *
from DrawGraph import *
from DirectedMyGraph import *
from Algorithms import *
# from DirectedDrawGraph import *

import random
import networkx as nx
import matplotlib.pyplot as plt

# === Zadanie 3: Generowanie silnie spójnego grafu z wagami ===

def generate_strongly_connected_digraph(n, p):
    while True:
        g_list = generate_random_directed_graph_by_probability(n, p)
        comp = kosaraju(g_list)
        if len(set(comp.values())) == 1:
            return g_list

def assign_random_weights(graph: DirectedAdjacencyList, low=-5, high=10):
    weights = {}
    for u in range(graph.n):
        for v in graph.adj[u]:
            weight = random.randint(low, high)
            weights[(u, v)] = weight
            graph.set_edge_weight(u, v, weight)  # teraz ta metoda istnieje
    return weights

def draw_weighted_digraph(graph: DirectedAdjacencyList, weights: dict):
    G = nx.DiGraph()
    for u in range(graph.n):
        for v in graph.adj[u]:
            G.add_edge(u, v, weight=weights[(u, v)])

    pos = nx.circular_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Silnie spójny digraf z wagami [-5, 10]")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Zadanie 1
    # Napisać program do kodowania grafów skierowanych (digrafów) i do generowania losowych digrafów z zespołu G(np).
    
    n = 7; p = 0.4

    # g_list = generate_random_directed_graph_by_probability(n, p)
    # print("Lista sąsiedztwa:")
    # print(g_list)
    # print()
    # g_mat = DirectedAdjacencyMatrix(g_list)
    # print("Macierz sąsiedztwa:")
    # print(g_mat)
    # print()
    # g_inc = DirectedIncidenceMatrix(g_list)
    # print("Macierz incydencji:")
    # print(g_inc)


    ## konwersja do NetworkX DiGraph

    # G = nx.DiGraph()
    # for u in range(g_list.n):
    #     for v in g_list.adj[u]:
    #         G.add_edge(u, v)

    # # ustawienie pozycji na okręgu
    # pos = nx.circular_layout(G)

    # # rysowanie
    # plt.figure(figsize=(6,6))
    # nx.draw(G, pos, with_labels=True, arrows=True, node_color="lightblue", edge_color="gray", 
    #         font_weight="bold", font_size=10)
    # plt.title(f"Digraf G({n}, {p})")
    # plt.axis("off")

    ## plt.show()

    # Zadanie 2
    # Zaimplementować algorytm Kosaraju do szukania silnie spójnych składowych na digrafie i zastosować go do digrafu losowego
    # Przykładowe użycie: wygeneruj losowy graf i zastosuj algorytm
    # g_list = generate_random_directed_graph_by_probability(n, p)

    # comp_map = kosaraju(g_list)

    # # # Grupowanie wierzchołków według numeru składowej
    # components: dict[int, list[int]] = {}
    # for vertex, cid in comp_map.items():
    #     components.setdefault(cid, []).append(vertex)
    # print()
    # print("Liczba silnie spójnych składowych:", len(components))
    # for cid, vertices in components.items():
    #     print(f"Składowa {cid}: {sorted(vertices)}")
    
    # plt.show()

    # Zadanie 3
    # Wykorzystując algorytmy z powyższych punktów wygenerować losowy
    # silnie spójny digraf. Łukom tego digrafu przypisać losowe wagi będące
    # liczbami całkowitymi z zakresu [−5, 10]. Zaimplementować algorytm
    # Bellmana-Forda do znajdowania najkrótszych ścieżek od danego wierz-
    # chołka.

    source = 0

    # 1. Generowanie silnie spójnego digrafu
    graph = generate_strongly_connected_digraph(n, p)
    weights = assign_random_weights(graph, -5, 10)

    # 2. Rysowanie
    # draw_weighted_digraph(graph, weights)

    # === Zadanie 3: Bellman-Ford ===
    print(f"\n[Zadanie 3] Najkrótsze ścieżki od wierzchołka {source}:")
    try:
        distances = bellman_ford(graph, weights, source)
        for v in range(n):
            print(f"  {source} → {v}: {distances[v]}")
    except ValueError as e:
        print("Błąd:", e)


    # Zadanie 4
    # Zaimplementować algorytm Johnsona do szukania odegłości pomiędzy
    # wszystkimi parami wierzchołków na ważonym grafie skierowanym.


    print(f"\n[Zadanie 4] Odległości pomiędzy wszystkimi parami (Johnson):")
    try:
        D = carl_johnson(graph)
        for u in range(n):
            for v in range(n):
                dist = D[u][v]
                out = f"{dist:.1f}" if dist < float('inf') else "inf"
                print(f"d({u},{v}) = {out}")
    except ValueError as e:
        print("Błąd:", e)

