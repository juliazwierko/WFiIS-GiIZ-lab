from MyGraph import *
from DrawGraph import * 
import networkx as nx
import random


def is_graphic_sequence(sequence: list[int]) -> bool:
    """
    Checks if the given degree sequence is graphic
    (if there exists a simple graph with such vertex degrees).
    """
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
    """
    Creates a simple graph from a graphic sequence.
    """
    if not is_graphic_sequence(sequence):
        raise ValueError("The given sequence is not graphic")
    
    print("· The given sequence is graphic")

    size = len(sequence)
    data = [[] for _ in range(size)]
    nodes = sorted([(degree, i) for i, degree in enumerate(sequence)], reverse=True)
    
    while nodes and nodes[0][0] > 0:
        degree, node = nodes.pop(0)
        nodes.sort(reverse=True)
        for i in range(degree):
            neighbor_degree, neighbor = nodes[i]
            data[node].append(neighbor)
            data[neighbor].append(node)
            nodes[i] = (neighbor_degree - 1, neighbor)
        nodes.sort(reverse=True)
    
    return AdjacencyList(size, data, sum(sequence) // 2)


def randomize_graph(graph: AdjacencyList, iterations: int = 100):
    """
    Randomizes the graph by swapping random pairs of edges.
    """
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
    """
    Finds the largest connected component in the graph.
    """
    G = nx.Graph()
    for node, neighbors in enumerate(graph.data):
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    largest_component = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_component)
    
    new_data = [[] for _ in range(len(largest_component))]
    mapping = {node: i for i, node in enumerate(sorted(largest_component))}
    for node in largest_component:
        for neighbor in G[node]:
            new_data[mapping[node]].append(mapping[neighbor])
    
    return AdjacencyList(len(largest_component), new_data, subgraph.number_of_edges())


if __name__ == "__main__":
    # Zadanie 1
    # Ssprawdzenie, czy dana sekwencja liczb naturalnych
    # jest ciągiem graficznym, i do konstruowania grafu prostego o stopniach
    # wierzchołków zadanych przez ciąg graficzny.

    # sequence = [3, 3, 2, 2, 2, 1]  # not graphic
    sequence = [3, 3, 2, 2, 1, 1]  # graphic

    graph = None

    if is_graphic_sequence(sequence):
        graph = construct_graph_from_sequence(sequence)
        print("Graph created from the graphic sequence:")
        print(graph)
        Draw(graph, "graph_from_sequence", "Graph from Graphic Sequence")
    else:
        print("The given sequence is not graphic")

    # Zadanie 2
    # zamienia losowo wybraną parę krawędzi: ab i cd na parę ad i bc.
    if graph is not None:
        randomize_graph(graph, 50)
        print("Graph after randomization:")
        print(graph)
        Draw(graph, "randomized_graph", "Randomized Graph")
    else:
        print("Cannot randomize the graph.")

    # Zadanie 3: 
    # Napisać program do znajdowania największej spójnej składowej na grafie.
    if graph is not None:
        largest_component_graph = largest_connected_component(graph)
        print("Largest connected component of the graph:")
        print(largest_component_graph)
        Draw(largest_component_graph, "largest_component", "Largest Connected Component")
    else:
        print("Cannot find the largest connected component.")