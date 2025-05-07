import random
import numpy as np
from typing import Union
from enum import Enum, auto
import random
import copy
import heapq

# -----------------------------  Zestaw 1  -----------------------------------

# GraphRepresentationType - dziedziczy po Enum.
# Będzie reprezentować różne sposoby przedstawiania grafu.
class GraphRepresentationType(Enum):
    AdjacencyList = auto()     
    AdjacencyMatrix = auto()   
    IncidenceMatrix = auto()   


# Deklaracja klasy bazowej Graph, która będzie wspólna dla różnych 
# reprezentacji grafów. Zawiera podstawowe właściwości i metody.
class Graph:
    size: int                               # liczba wierzchołków w grafie
    data: list[list[int]]                   # struktura grafu w postaci listy list liczb całkowitych. 
    edges: Union[int, None]                 # liczba krawędzi w Grafie
    type: GraphRepresentationType           # określa typ reprezentacji grafu na podstawie enuma GraphRepresentationType (linijka 13)
    weights: dict[tuple[int, int], int]     # mapowanie par wierzchołkow na ich wage - zostało dodano w trakcie wykonywania zestawu 3

    # Konstruktor klasy Graph
    def __init__(self, size: int, data: list[list[int]], edges: int, weights: dict[tuple[int, int], int] = None):
        self.data = data
        self.size = size
        self.edges = edges
        self.weights = weights

    # Reprezentacja tekstowa obiektów klasy Graph
    def __str__(self):
        return str(self.data)
    
    # Dodanie losowych (lub określonych) wag do krawędzi grafu, używając funkcji func() jako generatora wag.
    def add_weights(self, func):
        self.weights = {}                       # reseting old weights
        if self.type != GraphRepresentationType.AdjacencyList:
            graph = self.to_AL()
        else:
            graph = self
        for node, neighbors in enumerate(graph.data):
            for neighbor in neighbors:
                if node < neighbor:             # avoid double counting     
                    if (node, neighbor) not in self.weights.keys():
                        weight = func()
                        self.weights[(node, neighbor)] = weight
        self.weights  = dict(sorted([(edge, weight) for edge, weight in self.weights.items()], key = lambda x: x[0][0]))            # easier representation when sorted

class AdjacencyMatrix(Graph): pass
class AdjacencyList(Graph): pass
class IncidenceMatrix(Graph): pass

class AdjacencyList(Graph):
    # Deklaracja stalej klasy
    type: GraphRepresentationType = GraphRepresentationType.AdjacencyList

    # Konstruktor - wywoluje konstruktor klasy bazowej Graph
    def __init__(
        self,size: int,
        data: list[list[int]],
        edges: int, 
        weights: Union[None, dict[tuple[int, int], int]] = None,
    ):
        super().__init__(size, data, edges, weights)

    # Reprezentacja tekstowa:
    # N - liczba wierzcholkow; L - liczba krawedzi
    # Zapis postaci: wierzcholek: sąsiedzi
    def __str__(self):
        res = f"N: {self.size} L: {self.edges}\n"
        for index, neighbors in enumerate(self.data):
            res += f"{index}: {neighbors}\n"
        return res

    # Tworzenie grafu na podstawie pliku tekstowego
    @classmethod
    def from_txt(cls, lines: list[str]) -> "AdjacencyList":
        data = []
        edges = 0
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split(":")
            neighbors = list(map(int, parts[1].strip().split()))
            neighbors = [n - 1 for n in neighbors]  # przekształcamy z 1-based na 0-based
            data.append(neighbors)
            edges += len(neighbors)

        edges //= 2  # graf nieskierowany
        size = len(data)
        return cls(size, data, edges)

    # Konwertuje graf z listy sąsiedztwa na macierz sąsiedztwa.
    # n1 - kazdy wierzcholek, row - lista jego sąsiadów
    def to_AM(self) -> AdjacencyMatrix:
        new_data = np.zeros((self.size, self.size), dtype=int)
        for n1, row in enumerate(self.data):
            new_data[n1, row] = 1
        return AdjacencyMatrix(self.size, new_data.tolist(), self.edges, copy.deepcopy(self.weights))

    # Konwertuje listę sąsiedztwa na macierz incydencji
    # n1 - kazdy wierzcholek, n2 - kazdy jego sąsiad
    def to_IM(self) -> IncidenceMatrix:
        new_data = np.zeros((self.size, self.edges), dtype=int)
        edges = 0  
        for n1, neighbors in enumerate(self.data):
            for n2 in neighbors:
                if n1 < n2: # nie dodajemy tej samej krawedzi dwa razy! np. (1,2) (2,1)
                    new_data[n1, edges] = 1  
                    new_data[n2, edges] = 1  
                    edges += 1  

        return IncidenceMatrix(self.size, new_data.tolist(), self.edges, copy.deepcopy(self.weights))

class AdjacencyMatrix(Graph):
    # Deklaracja stalej klasy
    type: GraphRepresentationType = GraphRepresentationType.AdjacencyMatrix

    # Konstruktor - wywoluje konstruktor klasy bazowej Graph
    def __init__(
        self,
        size: int,
        data: list[list[int]],
        edges: int,
        weights: Union[None, dict[tuple[int, int], int]] = None,
    ):
        super().__init__(size, data, edges, weights)

    # Reprezentacja tekstowa:
    # N - liczba wierzcholkow; L - liczba krawedzi
    # Zapis postaci: macierz o wartosciach 0/1
    def __str__(self):
        res = f"N: {self.size} L: {self.edges}\n"
        for neighbors in self.data:
            res += f"{neighbors}\n"
        return res

    @classmethod
    # Wczytuje graf z pliku tekstowego w formacie:
    def from_txt(cls, lines: list[str]) -> "AdjacencyMatrix":
        data = [list(map(int, line.strip().split())) for line in lines if line.strip()]
        size = len(data)
        edges = sum(sum(row) for row in data) // 2  # nieskierowany
        return cls(size, data, edges)

    # Konwersja do listy sąsiedztwa
    def to_AL(self) -> AdjacencyList:
        new_data = [[] for _ in range(self.size)]  

        for n1 in range(self.size):
            for n2 in range(self.size):
                if self.data[n1][n2]:  # Jeśli istnieje krawędź
                    new_data[n1].append(n2)

        return AdjacencyList(self.size, new_data, self.edges, copy.deepcopy(self.weights))

    # Konwersja do macierzy incydencji
    def to_IM(self) -> IncidenceMatrix:
        new_data = [[0 for _ in range(self.edges)] for _ in range(self.size)]
        edges = 0
        # iterujemy po wszystkich parach (n1, n2) w macierzy
        for n1, row in enumerate(self.data):
            for n2, val in enumerate(row):
                if n1 > n2 and val:
                    new_data[n1][edges] = 1
                    new_data[n2][edges] = 1
                    edges += 1
        if edges != self.edges:
            raise ValueError("")
        return IncidenceMatrix(self.size, new_data, self.edges, copy.deepcopy(self.weights))

# Reprezentacja grafu za pomocą macierzy incydencji
class IncidenceMatrix(Graph):
    # Deklaracja stalej klasy
    type: GraphRepresentationType = GraphRepresentationType.IncidenceMatrix
    
    # Konstruktor - wywoluje konstruktor klasy bazowej Graph
    def __init__(self, size: int, data: list[list[int]], edges: int, weights: Union[None, dict[tuple[int, int], int]] = None,):
        super().__init__(size, data, edges, weights)

    # Reprezentacja tekstowa:
    # N - liczba wierzcholkow; L - liczba krawedzi
    # Zapis postaci: macierz o wartosciach 0/1
    def __str__(self):
        res = f"N: {self.size} L: {self.edges}\n"
        for neighbors in self.data:
            res += f"{neighbors}\n"
        return res

    # Tworzenie macierzy incydencji na podstawie pliku tekstowego
    @classmethod   
    def from_txt(cls, lines: list[str]) -> "IncidenceMatrix":
        data = [list(map(int, line.strip().split())) for line in lines if line.strip()]
        size = len(data)
        edges = len(data[0]) if data else 0
        return cls(size, data, edges)

    # Konwersja do macierzy sąsiedstwa
    def to_AM(self) -> AdjacencyMatrix:
        new_data = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for edge in zip(*self.data): # iteracja po kolumnach, zip transponuje macierz, kazda iteracja to jedna kolumna
            nodes = [n for n, val in enumerate(edge) if val] # indeksy wierzcholkow kotre biora udzial w tej krawedzi
            if len(nodes) == 2:
                n1, n2 = nodes
                new_data[n1][n2] = 1
                new_data[n2][n1] = 1

        return AdjacencyMatrix(self.size, new_data, self.edges, copy.deepcopy(self.weights))

    # Konwersja do listy sąsiedstwa
    def to_AL(self) -> AdjacencyList:
        new_data = [[] for _ in range(self.size)] # lista sasiadow dla kazdego wierzcholka
        for edge in zip(*self.data):
            nodes = []
            for n, val in enumerate(edge):
                if val:
                    nodes.append(n)
            new_data[nodes[0]].append(nodes[1])
            new_data[nodes[1]].append(nodes[0])
        return AdjacencyList(self.size, new_data, self.edges, copy.deepcopy(self.weights))


# Generacja wszystkich możliwych krawędzi w grafie nieskierowanym dla danego rozmiaru grafu (liczby wierzchołków).
def complete_edges(size: int) -> list[tuple[int, int]]:
    return [(x, y) for x in range(size) for y in range(x + 1, size)]
 
# Generuje losowy graf nieskierowany o zadanej liczbie wierzchołków i krawędzi. 
# Wynikowy graf jest reprezentowany jako lista sąsiedztwa (Adjacency List).
def generate_random_graph_by_edges(size: int, edges: int) -> AdjacencyList:
    data = [[] for _ in range(size)]  
    edge_list = random.sample(complete_edges(size), edges)
    
    for n1, n2 in edge_list:
        data[n1].append(n2)
        data[n2].append(n1)

    return AdjacencyList(size, data, edges)

# Generuje losowy graf nieskierowany o zadanej liczbie wierzchołków oraz
# prawdopodobienstwa istnienia kazdej z mozliwych krawedzi. 
def random_graph_by_probability(size: int, probability: float) -> AdjacencyList:
    data = [[] for _ in range(size)]  
    edges = 0

    for n1, n2 in complete_edges(size):
        if random.random() <= probability:
            edges += 1
            data[n1].append(n2)
            data[n2].append(n1)

    return AdjacencyList(size, data, edges)

# -----------------------------  Zestaw 2  -----------------------------------

def is_graphic_sequence(sequence: list[int]) -> bool:
    sequence = sorted(sequence, reverse=True)
    while sequence and sequence[0] > 0:
        if all(x == 0 for x in sequence):
            return True
        if sequence[0] >= len(sequence) or any(x < 0 for x in sequence):
            return False
        first = sequence.pop(0)
        for i in range(first):
            sequence[i] -= 1
            if sequence[i] < 0:
                return False
        sequence.sort(reverse=True)
    return True

def construct_graph_from_sequence(sequence: list[int]) -> list[list[int]]:
    if not is_graphic_sequence(sequence):
        raise ValueError("The given sequence is not graphic")

    print("The given sequence is graphic")

    size = len(sequence)
    data = [[] for _ in range(size)]

    nodes = [(degree, i) for i, degree in enumerate(sequence)]

    while any(degree > 0 for degree, _ in nodes):
        nodes = [node for node in nodes if node[0] > 0]
        if not nodes:
            break

        nodes.sort(reverse=True)
        degree, node = nodes.pop(0)

        if degree > len(nodes):
            raise ValueError(f"Cannot assign {degree} edges to node {node}")

        for i in range(degree):
            neighbor_degree, neighbor = nodes[i]

            if node == neighbor:
                raise ValueError("Loop detected, invalid graph.")
            
            if neighbor in data[node]:
                raise ValueError("Multiple edges detected, invalid graph.")

            data[node].append(neighbor)
            data[neighbor].append(node)

            nodes[i] = (neighbor_degree - 1, neighbor)

    return data


def randomize_graph(graph: AdjacencyList, iterations: int = 100):
    edges = [(u, v) for u in range(graph.size) for v in graph.data[u] if u < v]
    for _ in range(iterations):
        if len(edges) < 2:
            break
        (a, b), (c, d) = random.sample(edges, 2)
        if b != c and d not in graph.data[a] and b not in graph.data[d]:
            graph.data[a].remove(b)
            graph.data[b].remove(a)
            graph.data[c].remove(d)
            graph.data[d].remove(c)
            graph.data[a].append(d)
            graph.data[d].append(a)
            graph.data[c].append(b)
            graph.data[b].append(c)
            edges.remove((a, b))
            edges.remove((c, d))
            edges.append((a, d))
            edges.append((c, b))


def largest_connected_component(graph: AdjacencyList):
    def dfs(start, visited, component):
        stack = [start]
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                component.append(node)
                for neighbor in graph.data[node]:
                    if not visited[neighbor]:
                        stack.append(neighbor)

    n = len(graph.data)
    visited = [False] * n
    components = []

    # Znajdź wszystkie spójne składowe
    for node in range(n):
        if not visited[node]:
            component = []
            dfs(node, visited, component)
            components.append(component)
    # Znajdź największą
    largest_component = max(components, key=len)
    largest_component.sort()

    mapping = {node: i for i, node in enumerate(largest_component)}

    new_data = [[] for _ in range(len(largest_component))]
    edge_set = set()

    for old_node in largest_component:
        new_node = mapping[old_node]
        for neighbor in graph.data[old_node]:
            if neighbor in mapping:
                new_neighbor = mapping[neighbor]
                new_data[new_node].append(new_neighbor)
                edge = tuple(sorted((new_node, new_neighbor)))
                edge_set.add(edge)

    edge_count = len(edge_set)

    return AdjacencyList(len(largest_component), new_data, edge_count)


# Zestaw 2 (4-6)

def generate_eulerian_graph(n: int) -> AdjacencyList:
    if n < 1:
        raise ValueError("Number of vertices must be positive")
    while True:
        seq = [random.randrange(2, n + 1 if n % 2 == 0 else n, 2) for _ in range(n)]
        if sum(seq) % 2 == 0 and is_graphic_sequence(seq):
            return construct_graph_from_sequence(seq)

def find_eulerian_cycle(graph: AdjacencyList) -> list[int]:
    g = copy.deepcopy(graph)
    cycle = []
    stack = [0]

    def count_components(g_copy: AdjacencyList) -> int:
        visited = [False] * g_copy.size

        def dfs(u):
            visited[u] = True
            for v in g_copy.data[u]:
                if not visited[v]:
                    dfs(v)

        comps = 0
        for u in range(g_copy.size):
            if not visited[u]:
                dfs(u)
                comps += 1
        return comps

    while stack:
        u = stack[-1]
        if g.data[u]:
            for v in list(g.data[u]):
                g.data[u].remove(v)
                g.data[v].remove(u)
                comps_after = count_components(g)
                g.data[u].append(v)
                g.data[v].append(u)
                if comps_after == 1 or len(g.data[u]) == 1:
                    g.data[u].remove(v)
                    g.data[v].remove(u)
                    stack.append(v)
                    break
        else:
            cycle.append(stack.pop())
    return cycle

def generate_k_regular_graph(n: int, k: int) -> AdjacencyList:
    if k >= n or (n * k) % 2 != 0:
        raise ValueError("Invalid parameters for k-regular graph")
    stubs = [v for v in range(n) for _ in range(k)]
    while True:
        random.shuffle(stubs)
        edges = set()
        simple = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or (u, v) in edges or (v, u) in edges:
                simple = False
                break
            edges.add((u, v))
        if not simple:
            continue
        data = [[] for _ in range(n)]
        for u, v in edges:
            data[u].append(v)
            data[v].append(u)
        return AdjacencyList(n, data, len(edges))

def find_hamiltonian_cycle(graph: AdjacencyList) -> list[int] | None:
    n = graph.size
    adj = graph.data
    visited = [False] * n
    path: list[int] = []

    def backtrack(u: int) -> bool:
        path.append(u)
        visited[u] = True
        if len(path) == n:
            if path[0] in adj[u]:
                path.append(path[0])
                return True
        else:
            for v in adj[u]:
                if not visited[v]:
                    if backtrack(v):
                        return True
        visited[u] = False
        path.pop()
        return False

    for start in range(n):
        if backtrack(start):
            return path
    return None

# -----------------------------  Zestaw 3  -----------------------------------

def generate_random_connected_graph(size: int, edges: int) -> AdjacencyList:
    random_graph = generate_random_graph_by_edges(size, edges)
    largest_connected_component_graph = largest_connected_component(random_graph)
    return largest_connected_component_graph


def relax(u:int, v:int, weights: dict[tuple[int, int], int] , d: dict[int, int], p: dict[int, Union[int, None]]):
    w = weights[(u, v)] if u < v else weights[(v, u)]     
    if d[v] > d[u] + w:
        d[v] = d[u] + w
        p[v] = u


def dijkstra(graph: Graph, start_node: int, print_paths: bool = False) -> tuple[dict[int, int], dict[int, list[int]]]: 
    if graph.type != GraphRepresentationType.AdjacencyList:
        graph = graph.to_AL()

    # inicjalizacja
    distances = {v: 1e10 for v in range(graph.size)}          
    predecessors = {v: None for v in range(graph.size)}       
    distances[start_node] = 0
    visited = set()                                           
    
    while len(visited) < graph.size:
        current_vertex = min((v for v in range(graph.size) if v not in visited), key=lambda v: distances[v]) 
        visited.add(current_vertex)                           # S = S ∪ {u}
        for neighbor in graph.data[current_vertex]:
            if neighbor not in visited:
                relax(current_vertex, neighbor, graph.weights, distances, predecessors)  
        #print(f"Visited: {visited}, Distances: {distances}, Predecessors: {predecessors}")      # Sprawdzenie działania algorytmu
    #print(f"Final Distances: {distances}, Predecessors: {predecessors}")  # Sprawdzenie działania algorytmu     disctances - ready distances, predecessors - only previous for each of vertices
    
    paths = {}
    for vertex in predecessors:
        path = [vertex]
        while predecessors[vertex] is not None:
            path.append( predecessors[vertex])
            vertex = predecessors[vertex]
        paths[path[0]] = path[::-1]
    
    if print_paths:
        print(f'Paths = {paths}')
        print(f'START: s = {start_node}')
        for i in range(graph.size):
            print(f'd({i})  = {distances[i]:3}   ==> [{" - ".join(str(v) for v in paths[i])}]') 
    
    return distances, paths


def get_distance_matrix(graph: Graph) -> np.ndarray:
    M = []
    print(M)
    for vertice in range(graph.size):
        distances, _ = dijkstra(graph, vertice)
        M.append([v for v in distances.values()])
    return M


def print_matrix_nicely(matrix: list[list[int]]) -> None:
    print(' ', end='   ')
    for i in range(len(matrix)):
        print(f'{i:2}', end = ' ')
    print()   
    print(' ', end='  ')
    for i in range(len(matrix)):
        print(f'---', end = '')
    print()
    for i,row in enumerate(matrix):
        print(f'{i:2}|', end = ' ')
        print(" ".join(f'{el:2}' for el in row))
    print()
    
    
def find_graph_center_and_minimax_based_on_distance_matrix(M: list[list[int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    M = np.array(M)
    sums = np.sum(M, axis=1)
    max_paths = np.max(M, axis=1)
    center = np.argmin(sums)
    minimax = np.argmin(max_paths)
    return (center, sums[center]), (minimax, max_paths[minimax])


def prim(graph: Graph, start_node: int) -> tuple[list[int], list[int], list[tuple[int, int, int]]]:
    from DrawGraph import Draw
    
    if graph.type != GraphRepresentationType.AdjacencyList:
        graph = graph.to_AL()
    
    T = [start_node]
    W = [i for i in range(graph.size) if i != start_node]
    mst = []
    edges = []
    
    for neighbor in graph.data[start_node]:
        weight = graph.weights[(start_node, neighbor)] if start_node < neighbor else graph.weights[(neighbor, start_node)]
        heapq.heappush(edges, (weight, start_node, neighbor))     
    
    i=0
    while len(T) < graph.size:
        i += 1
        # print(f'T = {T}')
        # print(f'W = {W}')
        # print(f'edges = {edges}')
        weight,u, v = heapq.heappop(edges)
        if v in W:
            mst.append((u, v, weight))
            T.append(v)
            W.remove(v)
            Draw(graph, filename=f"mst_step_{i}.png", legend_title=f"MST Step {i}", output_dir='outputs/03/mst_steps_prim', with_weights=True, mst=mst)
            
            for neighbor in graph.data[v]: 
                if neighbor in W:
                    weight = graph.weights[(v, neighbor)] if v<neighbor else graph.weights[(neighbor, v)]
                    heapq.heappush(edges, (weight, v, neighbor))

    return T, W, mst


def kruskal(graph: Graph) -> tuple[list[int], list[int], list[tuple[int, int, int]]]:
    
    from DrawGraph import Draw
    
    if graph.type != GraphRepresentationType.AdjacencyList:
        graph = graph.to_AL()
    
    edges = [(w,u,v) for (u,v), w in graph.weights.items()]
    edges.sort()
    parent = {v: v for v in range(graph.size)}  
    rank = {v: 0 for v in range(graph.size)}
    
    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]
    
    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            elif rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            else:
                parent[root_v] = root_u
                rank[root_u] += 1
                
    mst = []
    i = 0
    for w, u, v in edges:
        if find(u) != find(v):  
            union(u, v)       
            mst.append((u, v, w)) 
        i += 1
        Draw(graph, filename=f"kruskal_step_{i}.png", legend_title=f"Kruskal Step {i}", output_dir="outputs/03/mst_steps_kruskal", with_weights=True, mst=mst)
                
    return mst
