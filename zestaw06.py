import random
from DirectedMyGraph import *
from Algorithms import *

# Zadanie 1
# Zaimplementować algorytm PageRank dla digrafu. Zastosować dwie poniższe metody i porównać wyniki.
# (a) Metoda polegającą na przechodzeniu od wierzchołka do sąsiedniego wierzchołka za pomocą błądzenia przypadkowego z prawdopodobieństwem 1-d i teleportacji z prawdopodobieństwem d.
# Przyjąć d = 0.15. PageRank wyliczyć jako częstość odwiedzin danego wierzchołka.
# (b) Metoda iteracji wektora obsadzeń pt. Dla t = 0 przyjąć p0 =(1/n, ..., 1/n), a następnie powtarzać iteracyjnie obliczenie pt+1 = ptP,
# dla t = 1,2,... , gdzie P jest macierza stochastyczną postaci Pij =(1-d)Aij/di + d/n, a dj jest stopniem wyjściowym wierzchołka j,
# a Aij macierzą sąsiedztwa. PageRank wylicza się jako wartości elementów wektora obsadzeń po wielu interacjach.
# Jeżeli te wartości się zmieniają w czasie, to PageRank wylicza się jako średnie tych elementów.

# Przykładowe użycie:
n = 100   # liczba wierzchołków
p = 0.05  # prawdopodobieństwo istnienia krawędzi
graph: DirectedAdjacencyList = generate_random_directed_graph_by_probability(n, p)

# Zapewnienie, że każdy wierzchołek ma co najmniej jedną krawędź wychodzącą
for u in range(n):
    if graph.out_degree(u) == 0:
        v = random.randrange(n)
        while v == u:
            v = random.randrange(n)
        graph.add_edge(u, v)

pr_rw = pagerank_random_walk(graph)
pr_pi, iterations = pagerank_power_iteration(graph)

# Sortowanie i printowanie
ranking_rw = sorted(enumerate(pr_rw), key=lambda x: x[1], reverse=True)
ranking_pi = sorted(enumerate(pr_pi), key=lambda x: x[1], reverse=True)

print("Ranking PageRank (błądzenie losowe):")
for rank, (node, score) in enumerate(ranking_rw, 1):
    print(f"{rank}. Wierzchołek {node}: {score:.6f}")
print("\nRanking PageRank (iteracja potęgowa):")
for rank, (node, score) in enumerate(ranking_pi, 1):
    print(f"{rank}. Wierzchołek {node}: {score:.6f}")
print(f"\nLiczba iteracji do zbieżności (metoda potęgowa): {iterations}")
