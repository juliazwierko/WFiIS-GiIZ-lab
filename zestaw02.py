from MyGraph import *
from DrawGraph import * 

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
    # Zamień losowo wybraną parę krawędzi: ab i cd na parę ad i bc.
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