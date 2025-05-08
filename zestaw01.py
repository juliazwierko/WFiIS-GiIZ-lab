from MyGraph import *
from DrawGraph import * 

if __name__ == "__main__":
    # Zadanie 1:
    '''
    Napisać program kodujący grafy proste za pomocą macierzy sąsiedz-
    twa, macierzy incydencji i list sąsiędztwa. Stworzyć moduł do zmiany
    danego kodowania na pozostałe.
    '''
    # generacja dla grafow losowych
    print("----------------- GENERACJA LOSOWA -----------------")
    print("Przyklad 1(graf losowy) ---> jest przeksztalcany do listy sąsiedstwa na ktorej wykonujemy konwersje ") 
    graph_AdjacencyList = generate_random_graph_by_edges(5, 6) 
    graph_AdjacencyMatrix = graph_AdjacencyList.to_AM()         
    graph_IncidenceMatrix = graph_AdjacencyList.to_IM()    

    print("Lista sąsiedstwa")   
    print(graph_AdjacencyList)
    print("Macierz sąsiedstwa")
    print(graph_AdjacencyMatrix)
    print("Macierz incydencji")
    print(graph_IncidenceMatrix)

    print("\nPrzyklad 2(graf losowy) ---> macierzy incydencji na ktorej wykonujemy konwersje ") 
    print("Lista sąsiedstwa") 
    print(graph_IncidenceMatrix.to_AL())
    print("Macierz sąsiedstwa")
    print(graph_IncidenceMatrix.to_AM())  
    
    # generacja dla plikow tekstowych
    print("----------------- WCZYTYWANIE Z PLIKU -----------------")
    with open("inputs/01/lista.txt") as f: # lista sasiedstwa (przyklad wziety z "Przykładowe wejście – zestaw 1.")
        al = AdjacencyList.from_txt(f.readlines())
    print("Wczytana lista sąsiedztwa:\n", al)
    am =  al.to_AM()
    im = al.to_IM()
    print("\nZ lista sąsiedztwa na macierz:")
    print(am)
    print("\nZ lista sąsiedztwa na incydencję:")
    print(im)
    Draw(al, "al01", "graph from Adjacency List", "outputs/01")
    Draw(am, "am01", "graph from Adjacency Matrix", "outputs/01")
    Draw(im, "im01", "graph from Incidence Matrix", "outputs/01")


    with open("inputs/01/macierz.txt") as f: # macierz sasiedstwa (przyklad wziety z "Przykładowe wejście – zestaw 1.")
        am = AdjacencyMatrix.from_txt(f.readlines())
    print("\nWczytana macierz sąsiedztwa:\n", am)
    al = am.to_AL()
    im = am.to_IM()
    print("\nZ macierz na listę:")
    print(al)
    print("\nZ macierz na incydencję:")
    print(im)
    Draw(al, "al02", "graph from Adjacency List", "outputs/01")
    Draw(am, "am02", "graph from Adjacency Matrix", "outputs/01")
    Draw(im, "im02", "graph from Incidence Matrix", "outputs/01")

    with open("inputs/01/incydencja.txt") as f: # macierz incydencji (przyklad wziety z "Przykładowe wejście – zestaw 1.")
        im = IncidenceMatrix.from_txt(f.readlines())  
    print("\nWczytana macierz incydencji:\n", im)
    al = im.to_AL()
    am = im.to_AM()
    print("\nZ incydencja na listę:")
    print(al)
    print("\nZ incydencja na macierz sąsiedstwa:")
    print(am)
    Draw(al, "al03", "graph from Adjacency List", "outputs/01")
    Draw(am, "am03", "graph from Adjacency Matrix", "outputs/01")
    Draw(im, "im03", "graph from Incidence Matrix", "outputs/01")


    # Zadanie 2:
    '''
    Napisać program do wizualizacji grafów prostych używający reprezen-
    tacji, w której wierzchołki grafu są równomiernie rozłożone na okręgu.
    '''
    Draw(graph_AdjacencyList, "graph_AdjacencyList", "graph from Adjacency List", "outputs/01")
    Draw(graph_AdjacencyMatrix, "graph_AdjacencyMatrix", "graph from Adjacency Matrix", "outputs/01")
    Draw(graph_IncidenceMatrix, "graph_IncidenceMatrix", "graph from Incidence Matrix", "outputs/01")


    # Zadanie 3:
    '''
    Napisać program do generowania grafów losowych G(n, l) oraz G(n, p).
    '''
    # 1) G(n,l)
    random_graph1 = generate_random_graph_by_edges(7, 10)   
    print(random_graph1)
    Draw(random_graph1, "random_graph1", "7, 10 - random graph", "outputs/01")

    # 2) G(n,p)
    random_graph2 = random_graph_by_probability(7, 0.5)
    print(random_graph2)
    Draw(random_graph2, "random_graph2", "7, 0.5 - random graph", "outputs/01")