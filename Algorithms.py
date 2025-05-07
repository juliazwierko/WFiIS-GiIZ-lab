from DirectedMyGraph import *

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
