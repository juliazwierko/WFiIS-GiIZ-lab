from DirectedMyGraph import *

# Zestaw 4 (1-2) ------------------------------------------------------------
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
                # jeśli jednak nie ma wyjść (najcichszy "haczyki"), teleportujemy
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
        # Sprawdzenie zbieżności (norma L1 różnicy wektorów)
        diff = sum(abs(new_pr[i] - pr[i]) for i in range(n))
        pr = new_pr
        if diff < epsilon:
            return pr, it
    return pr, max_iter

