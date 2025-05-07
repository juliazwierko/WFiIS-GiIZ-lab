from MyGraph import *
from DrawGraph import * 

if __name__ == "__main__":
    # Zadanie 1
    # Sprawdzenie, czy dana sekwencja liczb naturalnych
    # jest ciągiem graficznym, i do konstruowania grafu prostego o stopniach
    # wierzchołków zadanych przez ciąg graficzny.


    # sequence = [3, 3, 2, 2, 1, 1]  # graphic

    # https://d3gt.com/unit.html?graphic-sequence
    # sequence = [4,4,3,1,2]  # not graphic
    # sequence = [4, 2, 2, 3, 2, 1, 4, 2, 2, 2, 2]
    # sequence = [3, 3, 2, 2, 2, 1]  # not graphic
    sequence = [5, 3, 3, 3, 2, 2, 0]
    

    graph = None

    if is_graphic_sequence(sequence):
        try:
            adjacency_data = construct_graph_from_sequence(sequence)
            edge_count = sum(len(neighbors) for neighbors in adjacency_data) // 2
            graph = AdjacencyList(len(sequence), adjacency_data, edge_count)

            print("Graph created from the graphic sequence:")
            print(graph)
            Draw(graph, "graph_from_sequence", "Graph from Graphic Sequence")

        except ValueError as e:
            print("Error while constructing the graph:", e)

    else:
        print("The given sequence is not graphic")

    # # Zadanie 2
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


    # Zadanie 4:
    # Używając powyższych programów napisać program do tworzenia losowego grafu eulerowskiego i znajdowania na nim cyklu Eulera.
    n = 8
    euler_graph = generate_eulerian_graph(n)
    print("Generated Eulerian graph (degrees):", [len(nei) for nei in euler_graph.data])
    euler_cycle = find_eulerian_cycle(euler_graph)
    print("Eulerian cycle:", euler_cycle)
    Draw(euler_graph, "eulerian_graph.png", "Eulerian Graph")

    # Zadanie 5:
    # Napisać program do generowania losowych grafów k-regularnych.
    n, k = 7, 2
    k_reg_graph = generate_k_regular_graph(n, k)
    print(f"Generated {k}-regular graph:")
    print(k_reg_graph)
    Draw(k_reg_graph, "k_regular_graph.png", f"{k}-Regular Graph")

    # Zadanie 6:
    # Napisać program do sprawdzania (dla małych grafów), czy graf jest hamiltonowski.
    small_graph = random_graph_by_probability(6, 0.6)
    print("Small graph for Hamiltonian check:")
    print(small_graph)
    ham_cycle = find_hamiltonian_cycle(small_graph)
    if ham_cycle:
        print("Hamiltonian cycle found:", ham_cycle)
        Draw(small_graph, "hamiltonian_graph.png", "Hamiltonian Graph")
    else:
        print("No Hamiltonian cycle exists for this graph.")

