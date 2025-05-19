from DirectedMyGraph import *
from collections import deque, defaultdict
from DrawGraph import *
import random


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

# Funkcja BFS, która szuka najkrótszej ścieżki powiększającej(p) w grafie rezydualnym
def bfs_with_trace(residual_graph, source, sink, parent):
    visited = set()             # Zbiór odwiedzonych wierzchołków
    queue = deque([source])     # Kolejka BFS, zaczynamy od źródła
    visited.add(source)
    parent[source] = None    

    while queue:
        u = queue.popleft()                                    # Bierzemy pierwszy wierzchołek z kolejki
        for v in residual_graph[u]:                            # Dla każdego sąsiada u
            if v not in visited and residual_graph[u][v] > 0:  # Jeśli nie był odwiedzony i jest przepustowość
                visited.add(v)
                parent[v] = u       # Zapamiętujemy ścieżkę
                queue.append(v)
                if v == sink:      
                    return True, list(visited) 
    return False, list(visited)    

# Główna funkcja algorytmu Ford-Fulkerson z podejściem Edmondsa-Karpa + rysowanie iteracji
def ford_fulkerson_edmonds_karp_with_debug(graph, source, sink):
    residual_graph = defaultdict(dict)
    
    for (u, v), capacity in graph.weighted_edges.items():
        residual_graph[u][v] = capacity # Początkowa przepustowość
        if v not in residual_graph or u not in residual_graph[v]:
            residual_graph[v][u] = 0    # Krawędź wsteczna ma na start 0

    parent = {}          # Słownik śledzenia ścieżki
    max_flow = 0         # Początkowy przepływ
    step_counter = 1   

    # Startowy graf
    Draw_Residual_Network(graph=graph,residual_graph=residual_graph,legend_title=f"Step {step_counter}: Initial Residual Graph",filename=f"step_{step_counter}.png")
    step_counter += 1

    # Główna pętla — szukamy ścieżek powiększających
    while True:
        # Używamy BFS do znalezienia ścieżki powiększającej (czyli takiej, po której można 
        # coś jeszcze przesłać). Jeśli nie znajdziemy takiej ścieżki, to kończymy.
        found, visited_nodes = bfs_with_trace(residual_graph, source, sink, parent)

        if not found:
            Draw_Residual_Network(graph=graph,residual_graph=residual_graph,legend_title=f"Step {step_counter}: No more augmenting paths",filename=f"step_{step_counter}.png",visited_nodes=visited_nodes)
            break
        
        # Idziemy od końca do początku i szukamy minimalnej dostępnej przepustowości
        # Sprawdzamy każdą krawędź na ścieżce od końca do początku i wybieramy najmniejszą 
        # przepustowość — bo więcej niż to nie da się przesłać.
        path_flow = float('inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, residual_graph[parent[s]][s]) # residual_graph[parent[s]][s] to dostępna przepustowość tej krawędzi.
            s = parent[s]
        
        # Aktualizujemy przepływy w grafie rezydualnym
        v = sink
        while v != source:
            u = parent[v]
            residual_graph[u][v] -= path_flow   # Przepustowość do przodu zmniejszamy
            residual_graph[v][u] += path_flow   # Przepustowość wstecz zwiększamy
            v = parent[v]

        # Wypisanie znalezionej ścieżki
        augmenting_path = []
        v = sink
        while v is not None:
            augmenting_path.append(v)
            v = parent[v]
        augmenting_path = list(reversed(augmenting_path))
        print(f"Step {step_counter}: Found augmenting path: {' -> '.join(map(str, augmenting_path))} with flow = {path_flow}")

        max_flow += path_flow # Zwiększamy całkowity przepływ
        parent.clear()        # Czyścimy rodziców na kolejne szukanie
        Draw_Residual_Network(graph=graph, residual_graph=residual_graph, legend_title=f"Step {step_counter}: After path, flow += {path_flow}",filename=f"step_{step_counter}.png",visited_nodes=visited_nodes)
        step_counter += 1

    return max_flow, residual_graph

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

