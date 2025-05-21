from MyGraph import *
from DrawGraph import *
from DirectedMyGraph import *
from Algorithms import *

import random
import networkx as nx
import matplotlib.pyplot as plt

# === Zadanie 3: Generowanie silnie spójnego grafu z wagami ===

def assign_random_weights(graph: DirectedAdjacencyList, low=-5, high=10):
    weights = {}
    for u in range(graph.n):
        for v in graph.adj[u]:
            weight = random.randint(low, high)
            graph.weighted_edges[(u, v)] = weight
            weights[(u, v)] = weight
    return weights

def draw_weighted_digraph(graph: DirectedAdjacencyList, weights: dict, filename="digraph.png"):
    """Rysuje digraf z wagami i zapisuje go do katalogu outputs/04/"""
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
    output_dir = "outputs/04"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Zadanie 1
    # Napisać program do kodowania grafów skierowanych (digrafów) i do generowania losowych digrafów z zespołu G(np).
    
    # n = 7; p = 0.4

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


    # === Zadanie 3 ===
    # Wykorzystując algorytmy z powyższych punktów wygenerować losowy
    # silnie spójny digraf. Łukom tego digrafu przypisać losowe wagi będące
    # liczbami całkowitymi z zakresu [−5, 10]. Zaimplementować algorytm
    # Bellmana-Forda do znajdowania najkrótszych ścieżek od danego wierzchołka.

    n = 7
    p = 0.2
    source = 0

    print("[Zadanie 3] Generowanie silnie spójnego digrafu bez cyklu ujemnego...")
    attempt = 0
    while True:
        attempt += 1
        print(f"\nPróba {attempt}:")
        # Generujemy losowy digraf
        graph = generate_random_directed_graph_by_probability(n, p)
        
        # Sprawdzamy silną spójność
        comp_map = kosaraju(graph)
        if len(set(comp_map.values())) == 1:
            weights = assign_random_weights(graph, -5, 10)
            
            # Debugowanie: sprawdzamy zgodność wag
            print("\nLista sąsiedztwa grafu:")
            print(graph)
            print("\nWagi krawędzi (z weights):")
            for (u, v), w in sorted(weights.items()):
                print(f"  {u} -> {v}: {w}")
            print("\nWagi krawędzi (z graph.weighted_edges):")
            for (u, v), w in sorted(graph.weighted_edges.items()):
                print(f"  {u} -> {v}: {w}")
            
            # Rozszerzony Bellman-Ford z poprzednikami
            n = graph.n
            dist = [float('inf')] * n
            dist[source] = 0
            predecessor = [None] * n
            for _ in range(n - 1):
                for u in range(n):
                    for v in graph.adj[u]:
                        if dist[u] != float('inf') and dist[u] + weights[(u, v)] < dist[v]:
                            dist[v] = dist[u] + weights[(u, v)]
                            predecessor[v] = u
            # Sprawdzenie cykli ujemnych
            cycle_detected = False
            for u in range(n):
                for v in graph.adj[u]:
                    if dist[u] != float('inf') and dist[u] + weights[(u, v)] < dist[v]:
                        print("Wygenerowany graf zawiera cykl o ujemnej wadze — losuję ponownie...")
                        cycle_detected = True
                        break
                if cycle_detected:
                    break
            if not cycle_detected:
                print("\nZnaleziono graf silnie spójny bez cykli ujemnych!")
                break
        else:
            print("Wygenerowany graf nie jest silnie spójny — losuję ponownie...")

    # Wypisujemy odległości i ścieżki
    print(f"\nNajkrótsze ścieżki od wierzchołka {source}:")
    for v in range(n):
        dist_value = dist[v] if dist[v] != float('inf') else "inf"
        print(f"  {source} -> {v}: {dist_value}")
        if dist_value != "inf" and v != source:
            path = []
            path_weight = 0
            current = v
            while current is not None:
                path.append(current)
                if predecessor[current] is not None:
                    path_weight += weights.get((predecessor[current], current), 0)
                current = predecessor[current]
            path.reverse()
            if path[0] == source:
                print(f"    Ścieżka: {' -> '.join(map(str, path))}")
                print(f"    Suma wag na ścieżce: {path_weight}")
            else:
                print(f"    Ścieżka: (brak ścieżki, wierzchołek nieosiągalny)")

    draw_weighted_digraph(graph, weights)

    # === Zadanie 4 ===
    # Zaimplementować algorytm Johnsona do szukania odegłości pomiędzy
    # wszystkimi parami wierzchołków na ważonym grafie skierowanym.


    print(f"\n[Zadanie 4] Odległości pomiędzy wszystkimi parami (Johnson):")
    try:
        D = carl_johnson(graph)
        # Określamy szerokość kolumny na podstawie maksymalnej długości wartości
        max_width = max(5, max(len(f"{D[u][v]:.1f}") for u in range(n) for v in range(n) if D[u][v] != float('inf')))
        max_width = max(max_width, len("inf"))  # Uwzględniamy "inf"

        # Nagłówek wiersza z numerami wierzchołków
        print("     ", end="")
        for v in range(n):
            print(f"{v:>{max_width}} ", end="")
        print("\n" + "-" * (6 + n * (max_width + 1)))

        # Wypisujemy macierz
        for u in range(n):
            print(f"{u:>2} | ", end="")
            for v in range(n):
                value = D[u][v]
                out = f"{value:.0f}" if value != float('inf') else "inf"
                print(f"{out:>{max_width}} ", end="")
            print()
    except ValueError as e:
        print("Błąd:", e)