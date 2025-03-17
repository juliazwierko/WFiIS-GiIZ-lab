import random
import numpy as np
from typing import Union, List 
from enum import Enum, auto


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

    def __init__(self, size: int, data: list[list[int]], edges: int,):
        self.data = data
        self.size = size
        self.edges = edges

    def __str__(self):
        return str(self.data)

class AdjacencyMatrix(Graph): pass
class AdjacencyList(Graph): pass
class IncidenceMatrix(Graph): pass

class AdjacencyList(Graph):
    type: GraphRepresentationType = GraphRepresentationType.AdjacencyList

    def __init__(self,size: int, data: list[list[int]], edges: int,):
        super().__init__(size, data, edges)

    def __str__(self):
        res = f"N: {self.size} L: {self.edges}\n"
        for index, neighbors in enumerate(self.data):
            res += f"{index}: {neighbors}\n"
        return res

    def to_AM(self) -> AdjacencyMatrix:
        new_data = np.zeros((self.size, self.size), dtype=int)
        for n1, row in enumerate(self.data):
            new_data[n1, row] = 1
        return AdjacencyMatrix(self.size, new_data.tolist(), self.edges)

    def to_IM(self) -> IncidenceMatrix:
        new_data = np.zeros((self.size, self.edges), dtype=int)
        edges = 0  
        for n1, neighbors in enumerate(self.data):
            for n2 in neighbors:
                if n1 < n2:  
                    new_data[n1, edges] = 1  
                    new_data[n2, edges] = 1  
                    edges += 1  

        return IncidenceMatrix(self.size, new_data.tolist(), self.edges)

class AdjacencyMatrix(Graph):
    type: GraphRepresentationType = GraphRepresentationType.AdjacencyMatrix

    def __init__(
        self,
        size: int,
        data: list[list[int]],
        edges: int,
    ):
        super().__init__(size, data, edges)

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

        return AdjacencyList(self.size, new_data, self.edges)

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
        return IncidenceMatrix(self.size, new_data, self.edges)


class IncidenceMatrix(Graph):
    type: GraphRepresentationType = GraphRepresentationType.IncidenceMatrix
    
    def __init__(self, size: int, data: list[list[int]], edges: int,
    ):
        super().__init__(size, data, edges)

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

        return AdjacencyMatrix(self.size, new_data, self.edges)

    def to_AL(self) -> AdjacencyList:
        new_data = [[] for _ in range(self.size)]
        for edge in zip(*self.data):
            nodes = []
            for n, val in enumerate(edge):
                if val:
                    nodes.append(n)
            new_data[nodes[0]].append(nodes[1])
            new_data[nodes[1]].append(nodes[0])
        return AdjacencyList(self.size, new_data, self.edges)


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
