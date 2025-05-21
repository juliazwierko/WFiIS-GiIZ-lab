import random
from DirectedMyGraph import *
from Algorithms import *
from numba import njit

# Zadanie 1
# Zaimplementować algorytm PageRank dla digrafu. Zastosować dwie poniższe metody i porównać wyniki.
# (a) Metoda polegającą na przechodzeniu od wierzchołka do sąsiedniego wierzchołka za pomocą błądzenia przypadkowego z prawdopodobieństwem 1-d i teleportacji z prawdopodobieństwem d.
# Przyjąć d = 0.15. PageRank wyliczyć jako częstość odwiedzin danego wierzchołka.
# (b) Metoda iteracji wektora obsadzeń pt. Dla t = 0 przyjąć p0 =(1/n, ..., 1/n), a następnie powtarzać iteracyjnie obliczenie pt+1 = ptP,
# dla t = 1,2,... , gdzie P jest macierza stochastyczną postaci Pij =(1-d)Aij/di + d/n, a dj jest stopniem wyjściowym wierzchołka j,
# a Aij macierzą sąsiedztwa. PageRank wylicza się jako wartości elementów wektora obsadzeń po wielu interacjach.
# Jeżeli te wartości się zmieniają w czasie, to PageRank wylicza się jako średnie tych elementów.

# # Przykładowe użycie:
n = 100   # liczba wierzchołków
p = 0.05  # prawdopodobieństwo istnienia krawędzi
# graph: DirectedAdjacencyList = generate_random_directed_graph_by_probability(n, p)

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


# Zadanie 2
# Zaimplementować algorytm do wyszukiwania możliwie najkrótszej zamkniętej
# drogi przechodzącej przez wszystkie zadane wierzchołki rozrzucone
# na planszy kwadratowej. 
# Zastosować metodę symulowanego wyżarzania opartą o łańcuch Markowa, 
# którego pojedyncze kroki są wykonywane jako operacje 2-opt zgodnie 
# z algorytmem Metropolisa-Hastingsa.

coords = load_coordinates("inputs/xqf131.dat")

# ------- ACHTUNG <- heavy code below -------

# # Parametry do testowania
# outer_iter_values = [1100]
# inner_iter_values = [120000]
# num_runs = 10

# # Przechowywanie wyników
# all_costs = []
# all_routes = []
# all_histories = []

# for outer in outer_iter_values:
#     for inner in inner_iter_values:
#         for run in range(num_runs):
#             print(f"\n[INFO] Próba {run + 1}/{num_runs} z parametrami: outer={outer}, inner={inner}")
#             route, cost, history, last_temp, iters = simulated_annealing(
#                 coords, max_outer_iter=outer, max_inner_iter=inner
#             )
#             print(f"  Final cost: {cost:.2f}, Iterations: {iters}, Last temp: {last_temp:.9f}")

#             all_costs.append(cost)
#             all_routes.append(route)
#             all_histories.append(history)

# # Analiza wyników
# min_idx = np.argmin(all_costs)
# max_idx = np.argmax(all_costs)
# avg_cost = np.mean(all_costs)

# print("\n=== Podsumowanie wyników ===")
# print(f"Najlepszy koszt: {all_costs[min_idx]:.2f}")
# print(f"Najgorszy koszt: {all_costs[max_idx]:.2f}")
# print(f"Średni koszt z {num_runs} prób: {avg_cost:.2f}")

# # Zapisanie wykresów dla najlepszego rozwiązania
# plot_route(all_routes[min_idx], coords, all_costs[min_idx], filename="best_route.png")
# plot_cost_history(all_histories[min_idx], filename="best_cost_history.png")