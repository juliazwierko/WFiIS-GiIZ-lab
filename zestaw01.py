from MyGraph import *
from DrawGraph import * 

if __name__ == "__main__":
    # Zadanie 1
    # kodowanie grafow prostych za pomoca:
    # macierzy sąsiedstwa, macierzy incydencji, listy sasiedstwa
    # uzycie modulu = zmiana z jednego kodowania na pozostale
    graph_AdjacencyList = generate_random_graph_by_edges(5, 10) 
    graph_AdjacencyMatrix = graph_AdjacencyList.to_AM()         
    graph_IncidenceMatrix = graph_AdjacencyList.to_IM()        
    print(graph_AdjacencyList)
    print(graph_AdjacencyMatrix)
    print(graph_IncidenceMatrix)

    print(graph_AdjacencyList.to_AM())
    print(graph_AdjacencyList.to_IM())

    print(graph_IncidenceMatrix.to_AL())
    print(graph_IncidenceMatrix.to_AM())  
  

    # Zadanie 2
    # wizualizacja grafow, w ktorej wierzcholki sa rownomiernie
    # rozlozone po okregu
    Draw(graph_AdjacencyList, "graph_AdjacencyList", "graph from Adjacency List")
    Draw(graph_AdjacencyMatrix, "graph_AdjacencyMatrix", "graph from Adjacency Matrix")
    Draw(graph_IncidenceMatrix, "graph_IncidenceMatrix", "graph from Incidence Matrix")


    # Zadanie 3
    # generowanie grafa losowych
    # 1) G(n,l)
    # 2) G(n,p)
    random_graph1 = generate_random_graph_by_edges(7, 10)   
    print(random_graph1)
    Draw(random_graph1, "random_graph1", "7, 10 - random graph")

    random_graph2 = random_graph_by_probability(7, 0.5)
    print(random_graph2)
    Draw(random_graph2, "random_graph2", "7, 0.5 - random graph")