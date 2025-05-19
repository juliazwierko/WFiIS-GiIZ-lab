from DirectedMyGraph import *
from collections import deque
import random
import math
import numpy as np
from numba import njit
from numba.typed import List # bo numba nie lubi list innych niż numba.typed.List
import matplotlib.pyplot as plt
import os

# Zestaw 4  ------------------------------------------------------------
def kosaraju(graph: DirectedAdjacencyList) -> dict[int, int]:
    """
    Znajduje silnie spójne składowe w zadanym grafie skierowanym.
    
    Parametry:
        graph (DirectedAdjacencyList): graf skierowany reprezentowany listą sąsiedztwa.
    Zwraca:
        dict[int, int]: słownik mapujący wierzchołek na numer składowej (ID składowej).
    """
    # Przygotuj stos i słownik visited dla pierwszego DFS
    visited = [False] * graph.n
    stack: list[int] = []

    # Definicja DFS wypełniającego stos wierzchołkami wg czasu zakończenia
    def dfs_fill(v: int):
        visited[v] = True
        for u in graph.adj[v]:
            if not visited[u]:
                dfs_fill(u)
        stack.append(v)  # dodaj na stos po zakończeniu DFS z v

    # Wykonaj pierwszy DFS dla wszystkich wierzchołków
    for v in range(graph.n):
        if not visited[v]:
            dfs_fill(v)

    # Utwórz graf transponowany (odwróć krawędzie)
    transpose = DirectedAdjacencyList(graph.n)

    # Najpierw dodaj wszystkie wierzchołki (jeśli wymaga jawnego dodania)
    for u in range(graph.n):
        for v in graph.adj[u]:
            transpose.add_edge(v, u)  # odwórć krawędź u->v na v->u

    # Drugi DFS na grafie transponowanym: przetwarzaj wierzchołki ze stosu
    visited2 = [False] * graph.n
    component_map: dict[int, int] = {}
    comp_id = 0

    def dfs_assign(v: int):
        """DFS rekurencyjne oznaczające przynależność do bieżącej składowej."""
        visited2[v] = True
        component_map[v] = comp_id
        for u in transpose.adj[v]:
            if not visited2[u]:
                dfs_assign(u)

    # Przetwarzaj wierzchołki ze stosu (od ostatniego dodanego)
    while stack:
        v = stack.pop()
        if not visited2[v]:
            comp_id += 1
            dfs_assign(v)

    return component_map


def bellman_ford(g_list, weights, source):
    """
    Implementacja algorytmu Bellmana-Forda zgodnie z pseudokodem.
    Zwraca:
    - (False, _) jeśli istnieje cykl ujemny osiągalny ze źródła,
    - (True, ds) jeśli nie istnieje (gdzie ds[v] to najkrótsza ścieżka z s do v)
    """
    n = g_list.n
    dist = [float('inf')] * n
    dist[source] = 0

    for _ in range(n - 1):
        for u in range(n):
            for v in g_list.adj[u]:
                if dist[u] + weights[(u, v)] < dist[v]:
                    dist[v] = dist[u] + weights[(u, v)]

    for u in range(n):
        for v in g_list.adj[u]:
            if dist[u] + weights[(u, v)] < dist[v]:
                raise ValueError("Wykryto cykl o ujemnej wadze!")

    return dist

def add_s(graph: DirectedAdjacencyList) -> tuple[DirectedAdjacencyList, int, dict[tuple[int, int], int]]:
    """
    Dodaje sztuczny wierzchołek s połączony z wagą 0 z wszystkimi innymi.
    Zwraca nowy graf G0, numer nowego wierzchołka s oraz nową macierz wag.
    """
    G0 = graph.copy()
    n = G0.num_vertices()
    G0.n += 1
    G0.adj.append([])  # nowy wierzchołek s

    # skopiuj istniejące wagi
    w_new = dict(graph.weighted_edges)

    # dodaj krawędzie (s, v) z wagą 0
    for v in range(n):
        G0.adj[n].append(v)
        w_new[(n, v)] = 0

    return G0, n, w_new

def carl_johnson(graph: DirectedAdjacencyList) -> list[list[float]]:
    """
    Algorytm Johnsona (Nie mial na imie Carl) do znajdowania najkrótszych ścieżek między wszystkimi parami.
    Zwraca macierz odległości D.
    """
    G0, s, w = add_s(graph)  # krok 1

    ok, ds = bellman_ford(G0, w, s)  # krok 2
    if not ok:
        raise ValueError("Graf zawiera cykl o ujemnej wadze.")  # krok 3

    h = ds.copy()  # krok 6

    # krok 8–10: przeskalowanie wag
    w_reweighted = {}
    for (u, v), weight in graph.weighted_edges.items():
        w_reweighted[(u, v)] = weight + h[u] - h[v]

    # krok 11: macierz D
    n = graph.num_vertices()
    D = [[math.inf] * n for _ in range(n)]

    # krok 12–17: Dijkstra i przeskalowanie wyników
    for u in range(n):
        dist = dijkstra(graph, u, w_reweighted)  # krok 13
        for v in range(n):
            if dist[v] < math.inf:
                D[u][v] = dist[v] - h[u] + h[v]  # krok 15

    return D  # krok 18

# Zestaw 5 ------------------------------------------------------------
        
def get_layers_from_network(network: FlowNetwork, start_node: int = 0) -> list[list[int]]:
    visited = set([start_node])
    queue = deque([start_node])
    
    level = {start_node: 0}
    layers = {}
    
    while queue:
        node = queue.popleft()
        layers.setdefault(level[node], []).append(node)
        neigbours = [v for (u, v), _ in network.weighted_edges.items() if u == node]
        for v in neigbours:
            if v not in visited or level[v] > level[node] + 1:
                level[v] = level[node] + 1
                visited.add(v)
                queue.append(v)
    
    return [layers[i] for i in range(max(level.values()) + 1)]    

    
def generate_random_flow_network(nmbr_inter_layers: int, probability: float) -> FlowNetwork:
    
    network = FlowNetwork(20)
    prev_layer = [0]
    curr_vert = 0
    
    for _ in range(nmbr_inter_layers):
        new_layer = []
        remaining = prev_layer[:]
        
        for i in range(nmbr_inter_layers):                 
            if i < 2 or random.random() < probability:     
                curr_vert += 1
                node = None
                if not remaining:
                    node = random.choice(prev_layer)
                else:
                    node = random.choice(remaining)
                    remaining.remove(node)
                network.add_edge(node, curr_vert, func = lambda: random.randint(1,10))
                new_layer.append(curr_vert)
        if remaining:          
            for node in remaining:
                network.add_edge(node, random.choice(new_layer), func = lambda: random.randint(1,10))
        prev_layer = new_layer
     
    curr_vert += 1   
    for node in prev_layer:
        network.add_edge(node, curr_vert, func = lambda: random.randint(1,10))
    network.n = curr_vert + 1   
    network.network_layers = get_layers_from_network(network)
    network.refactor_adjacency_matrix(curr_vert + 1)
    # print(network)
    # print()
    
    for _ in range(2*nmbr_inter_layers):
        u = random.choice(range(1, curr_vert))    
        v = random.choice(range(1, curr_vert))
        added = False
        limit = 5 * nmbr_inter_layers                        
        while not added and limit > 0:
            if u != v and not network.edge_exists(u, v) and not network.edge_exists(v, u):
                network.add_edge(u, v, func = lambda: random.randint(1,10))
                added = True
            else:
                u = random.choice(range(1, curr_vert))
                v = random.choice(range(1, curr_vert))
                limit -= 1
        
        if limit <= 0 and not added:                       
            for u in range(1, curr_vert):
                for v in range(1, curr_vert):
                    if u != v and not network.edge_exists(u, v) and not network.edge_exists(v, u) and not added:
                        network.add_edge(u, v, func = lambda: random.randint(1,10))
                        added = True
        if not added:
            return network
        
    #print(network)
    return network


# Zestaw 6 (1) ------------------------------------------------------------

def pagerank_random_walk(graph: DirectedAdjacencyList, steps: int = 1_000_000, d: float = 0.15) -> list[float]:
    n = graph.num_vertices()
    visits = [0] * n
    current = random.randrange(n)  # losowy wierzchołek startowy
    for _ in range(steps):
        if random.random() < d:
            # teleportacja
            current = random.randrange(n)
        else:
            neighbors = list(graph.out_neighbors(current))
            if neighbors:
                current = random.choice(neighbors)
            else:
                # jeśli nie ma wyjść = teleportujemy
                current = random.randrange(n)
        visits[current] += 1
    # Normalizacja do sumy 1
    return [count / steps for count in visits]


def pagerank_power_iteration(graph: DirectedAdjacencyList,
                             d: float = 0.15, max_iter: int = 100, epsilon: float = 1e-8) -> tuple[list[float], int]:
    n = graph.num_vertices()
    pr = [1.0/n] * n
    for it in range(1, max_iter + 1):
        new_pr = [0.0] * n
        # Rozkład PageRank każdego wierzchołka na jego sąsiadów
        for u in range(n):
            neighbors = list(graph.out_neighbors(u))
            if neighbors:
                share = pr[u] / len(neighbors)
                for v in neighbors:
                    new_pr[v] += share
            else:
                # Jeśli wierzchołek nie ma wyjść, rozdzielamy jego wagę równomiernie (jak teleport do wszystkich)
                for v in range(n):
                    new_pr[v] += pr[u] / n
        # Dodanie teleportacji (część (1-d) do każdego wierzchołka)
        new_pr = [(1-d)/n + d * x for x in new_pr]
        # Sprawdzenie zbieżności
        diff = sum(abs(new_pr[i] - pr[i]) for i in range(n))
        pr = new_pr
        if diff < epsilon:
            return pr, it
    return pr, max_iter

# Zestaw 6 (2) ------------------------------------------------------------

def load_coordinates(filename):
    """Wczytaj współrzędne z pliku .dat"""
    with open(filename, 'r') as f:
        lines = f.readlines()
        coords = [list(map(float, line.strip().split())) for line in lines if line.strip()]
    return np.array(coords)

def initial_coordinates(coords, filename="initial_route.png"):
    route = np.arange(len(coords))  # trasa po kolei
    distance = total_distance(route, coords)
    plot_route(route, coords, distance, filename)
    print(f"Initial route distance: {distance:.2f}")

@njit
def euclidean(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

@njit
def total_distance(route, coords):
    dist = 0.0
    n = len(route)
    for i in range(n):
        a = coords[route[i]]
        b = coords[route[(i + 1) % n]]
        dist += euclidean(a, b)
    return dist

@njit
def two_opt_swap(route, i, k):
    new_route = route.copy()
    while i < k:
        temp = new_route[i]
        new_route[i] = new_route[k]
        new_route[k] = temp
        i += 1
        k -= 1
    return new_route

@njit
def simulated_annealing(coords, max_outer_iter=100, max_inner_iter=100):
    n = len(coords)
    current_route = np.arange(n)
    current_cost = total_distance(current_route, coords)

    best_route = current_route.copy()
    best_cost = current_cost

    cost_history = List()
    cost_history.append(current_cost)

    iters = 0
    temp = 0.0  # temp z ostatniej pętli

    for i in range(max_outer_iter, 0, -1):  # Pętla chłodzenia
        T = 0.001 * i * i
        temp = T

        for _ in range(max_inner_iter):  # Iteracje dla ustalonej T
            idx1 = np.random.randint(0, n)
            idx2 = np.random.randint(0, n)
            if idx1 == idx2 or abs(idx1 - idx2) < 2:
                continue

            i1 = min(idx1, idx2)
            i2 = max(idx1, idx2)

            new_route = two_opt_swap(current_route, i1, i2)
            new_cost = total_distance(new_route, coords)
            delta = new_cost - current_cost

            if delta < 0.0 or np.random.random() < np.exp(-delta / T):
                current_route = new_route
                current_cost = new_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_route = current_route.copy()

            cost_history.append(current_cost)
            iters += 1

    return best_route, best_cost, cost_history, temp, iters


def plot_route(route, coords, distance, filename="tsp.png"):
    ordered_coords = coords[route]
    closed_coords = np.vstack([ordered_coords, ordered_coords[0]])

    output_dir = "./outputs/06"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(8, 8))
    plt.plot(closed_coords[:, 0], closed_coords[:, 1], 'o-', color='blue')
    plt.title(f"Trasa (długość: {distance:.2f})")
    plt.xlim(np.min(coords[:, 0]) - 5, np.max(coords[:, 0]) + 5)
    plt.ylim(np.min(coords[:, 1]) - 5, np.max(coords[:, 1]) + 5)
    plt.xlabel("Położenie X")
    plt.ylabel("Położenie Y")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Zapisano wykres trasy do: {output_path}")


def plot_cost_history(cost_history, filename="cost.png"):
    output_dir = "./outputs/06"
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Zwykły wykres ─────────────────────────────
    output_path_linear = os.path.join(output_dir, filename)
    plt.figure(figsize=(8, 8))
    plt.plot(cost_history, color='black')
    plt.title("Historia kosztu")
    plt.xlabel("Iteracja")
    plt.ylabel("Koszt")
    plt.grid(True)
    plt.savefig(output_path_linear)
    plt.close()
    print(f"Zapisano wykres kosztu (liniowy) do: {output_path_linear}")
    
    # ── Logarytmiczna oś X ─────────────────────────
    base, ext = os.path.splitext(filename)
    output_path_log = os.path.join(output_dir, f"{base}_logx{ext}")
    plt.figure(figsize=(8, 8))
    plt.plot(cost_history, color='black')
    plt.xscale("log")
    plt.title("Historia kosztu (logarytmiczna oś X)")
    plt.xlabel("Iteracja (log)")
    plt.ylabel("Koszt")
    plt.grid(True, which="both")
    plt.savefig(output_path_log)
    plt.close()
    print(f"Zapisano wykres kosztu (log-x) do: {output_path_log}")