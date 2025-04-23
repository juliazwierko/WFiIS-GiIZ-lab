import random
import numpy as np
from typing import Union, List 
from enum import Enum, auto
# import networkx as nx
import random
import copy
# Zestaw 1

class GraphRepresentationType(Enum):
    """Typy reprezentacji grafu:
    1) AdjacencyList - Reprezentacja grafu jako lista sąsiedztwa.
    2) AdjacencyMatrix - Reprezentacja grafu jako macierz sąsiedztwa.
    3) IncidenceMatrix - Reprezentacja grafu jako macierz incydencji.
    """
    AdjacencyList = auto()      # Reprezentacja grafu jako lista sąsiedztwa
    AdjacencyMatrix = auto()    # Reprezentacja grafu jako macierz sąsiedztwa
    IncidenceMatrix = auto()    # Reprezentacja grafu jako macierz incydencji


class Graph:
    size: int
    data: list[list[int]]
    edges: Union[int, None]
    type: GraphRepresentationType
    weights: dict[tuple[int, int], int]

    def __init__(self, size: int, data: list[list[int]], edges: int, weights: dict[tuple[int, int], int] = None):
        self.data = data
        self.size = size
        self.edges = edges
        self.weights = weights

    def __str__(self):
        return str(self.data)
    
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
    type: GraphRepresentationType = GraphRepresentationType.AdjacencyList

    def __init__(
        self,size: int,
        data: list[list[int]],
        edges: int, 
        weights: Union[None, dict[tuple[int, int], int]] = None,
    ):
        super().__init__(size, data, edges, weights)

    def __str__(self):
        res = f"N: {self.size} L: {self.edges}\n"
        for index, neighbors in enumerate(self.data):
            res += f"{index}: {neighbors}\n"
        return res

    def to_AM(self) -> AdjacencyMatrix:
        new_data = np.zeros((self.size, self.size), dtype=int)
        for n1, row in enumerate(self.data):
            new_data[n1, row] = 1
        return AdjacencyMatrix(self.size, new_data.tolist(), self.edges, copy.deepcopy(self.weights))

    def to_IM(self) -> IncidenceMatrix:
        new_data = np.zeros((self.size, self.edges), dtype=int)
        edges = 0  
        for n1, neighbors in enumerate(self.data):
            for n2 in neighbors:
                if n1 < n2:  
                    new_data[n1, edges] = 1  
                    new_data[n2, edges] = 1  
                    edges += 1  

        return IncidenceMatrix(self.size, new_data.tolist(), self.edges, copy.deepcopy(self.weights))

class AdjacencyMatrix(Graph):
    type: GraphRepresentationType = GraphRepresentationType.AdjacencyMatrix

    def __init__(
        self,
        size: int,
        data: list[list[int]],
        edges: int,
        weights: Union[None, dict[tuple[int, int], int]] = None,
    ):
        super().__init__(size, data, edges, weights)

    def __str__(self):
        res = f"N: {self.size} L: {self.edges}\n"
        for neighbors in self.data:
            res += f"{neighbors}\n"
        return res

    def to_AL(self) -> AdjacencyList:
        new_data = [[] for _ in range(self.size)]  

        for n1 in range(self.size):
            for n2 in range(self.size):
                if self.data[n1][n2]:  # Jeśli istnieje krawędź
                    new_data[n1].append(n2)

        return AdjacencyList(self.size, new_data, self.edges, copy.deepcopy(self.weights))

    def to_IM(self) -> IncidenceMatrix:
        new_data = [[0 for _ in range(self.edges)] for _ in range(self.size)]
        edges = 0
        for n1, row in enumerate(self.data):
            for n2, val in enumerate(row):
                if n1 > n2 and val:
                    new_data[n1][edges] = 1
                    new_data[n2][edges] = 1
                    edges += 1
        if edges != self.edges:
            raise ValueError("")
        return IncidenceMatrix(self.size, new_data, self.edges, copy.deepcopy(self.weights))


class IncidenceMatrix(Graph):
    type: GraphRepresentationType = GraphRepresentationType.IncidenceMatrix
    
    def __init__(self, size: int, data: list[list[int]], edges: int, weights: Union[None, dict[tuple[int, int], int]] = None,):
        super().__init__(size, data, edges, weights)

    def __str__(self):
        res = f"N: {self.size} L: {self.edges}\n"
        for neighbors in self.data:
            res += f"{neighbors}\n"
        return res

    def to_AM(self) -> AdjacencyMatrix:
        new_data = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for edge in zip(*self.data):
            nodes = [n for n, val in enumerate(edge) if val]
            if len(nodes) == 2:
                n1, n2 = nodes
                new_data[n1][n2] = 1
                new_data[n2][n1] = 1

        return AdjacencyMatrix(self.size, new_data, self.edges, copy.deepcopy(self.weights))

    def to_AL(self) -> AdjacencyList:
        new_data = [[] for _ in range(self.size)]
        for edge in zip(*self.data):
            nodes = []
            for n, val in enumerate(edge):
                if val:
                    nodes.append(n)
            new_data[nodes[0]].append(nodes[1])
            new_data[nodes[1]].append(nodes[0])
        return AdjacencyList(self.size, new_data, self.edges, copy.deepcopy(self.weights))


def complete_edges(size: int) -> list[tuple[int, int]]:
    return [(x, y) for x in range(size) for y in range(x + 1, size)]
 
def generate_random_graph_by_edges(size: int, edges: int) -> AdjacencyList:
    data = [[] for _ in range(size)]  
    edge_list = random.sample(complete_edges(size), edges)
    
    for n1, n2 in edge_list:
        data[n1].append(n2)
        data[n2].append(n1)

    return AdjacencyList(size, data, edges)

def random_graph_by_probability(size: int, probability: float) -> AdjacencyList:
    data = [[] for _ in range(size)]  
    edges = 0

    for n1, n2 in complete_edges(size):
        if random.random() <= probability:
            edges += 1
            data[n1].append(n2)
            data[n2].append(n1)

    return AdjacencyList(size, data, edges)

# Zestaw 2

def is_graphic_sequence(sequence: list[int]) -> bool:
    sequence = sorted(sequence, reverse=True)
    while sequence and sequence[0] > 0:
        if sum(sequence) % 2 == 1 or sequence[0] >= len(sequence):
            return False
        first = sequence.pop(0)
        for i in range(first):
            sequence[i] -= 1
            if sequence[i] < 0:
                return False
        sequence.sort(reverse=True)
    return True

def construct_graph_from_sequence(sequence: list[int]) -> AdjacencyList:
    if not is_graphic_sequence(sequence):
        raise ValueError("· The given sequence is not graphic")
    
    print("· The given sequence is graphic")

    size = len(sequence)
    data = [[] for _ in range(size)]
    nodes = sorted([(degree, i) for i, degree in enumerate(sequence)], reverse=True)
    
    while nodes and nodes[0][0] > 0:
        degree, node = nodes.pop(0)
        nodes.sort(reverse=True)            # Po co sortujemy dwa razy?, tutaj jusz mamy posortowaną liste na wejściu
        for i in range(degree):
            neighbor_degree, neighbor = nodes[i]
            data[node].append(neighbor)
            data[neighbor].append(node)
            nodes[i] = (neighbor_degree - 1, neighbor)
        nodes.sort(reverse=True)            # tu jest
    
    return AdjacencyList(size, data, sum(sequence) // 2)


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

# def largest_connected_component(graph: AdjacencyList):
#     G = nx.Graph()
#     for node, neighbors in enumerate(graph.data):
#         for neighbor in neighbors:
#             G.add_edge(node, neighbor)
    
#     largest_component = max(nx.connected_components(G), key=len)
#     subgraph = G.subgraph(largest_component)
    
#     new_data = [[] for _ in range(len(largest_component))]
#     mapping = {node: i for i, node in enumerate(sorted(largest_component))}
#     for node in largest_component:
#         for neighbor in G[node]:
#             new_data[mapping[node]].append(mapping[neighbor])
    
#     return AdjacencyList(len(largest_component), new_data, subgraph.number_of_edges())


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


# Zestaw 3
def generate_random_connected_graph(size: int, edges: int) -> AdjacencyList:
    random_graph = generate_random_graph_by_edges(size, edges)
    largest_connected_component_graph = largest_connected_component(random_graph)
    return largest_connected_component_graph

