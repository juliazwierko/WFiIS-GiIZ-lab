import random

class DirectedAdjacencyList:
    """
    Prosty graf skierowany reprezentowany jako lista sąsiedztwa.
    """
    def __init__(self, n):
        self.n = n
        self.m = 0
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v):
        """
        Dodaje skierowany łuk u -> v (bez pętli i duplikatów).
        """
        if u == v:
            return
        if 0 <= u < self.n and 0 <= v < self.n:
            if v not in self.adj[u]:
                self.adj[u].append(v)
                self.m += 1

    def __str__(self):
        lines = []
        for u in range(self.n):
            if self.adj[u]:
                nbrs = ' '.join(str(v) for v in sorted(self.adj[u]))
                lines.append(f"{u}: {nbrs}")
            else:
                lines.append(f"{u}:")
        return '\n'.join(lines)

    def copy(self) -> "DirectedAdjacencyList":
        new = DirectedAdjacencyList(self.n)
        new.m = self.m
        # skopiuj listy sąsiedztwa
        new.adj = [nbrs.copy() for nbrs in self.adj]
        return new

class DirectedAdjacencyMatrix:
    """
    Prosty graf skierowany reprezentowany jako macierz sąsiedztwa.
    """
    def __init__(self, source):
        if isinstance(source, DirectedAdjacencyList):
            n = source.n
            self.n = n
            self.m = source.m
            self.matrix = [[0]*n for _ in range(n)]
            for u in range(n):
                for v in source.adj[u]:
                    self.matrix[u][v] = 1
        else:
            n = int(source)
            self.n = n
            self.m = 0
            self.matrix = [[0]*n for _ in range(n)]

    def __str__(self):
        lines = []
        for u in range(self.n):
            lines.append(' '.join(str(self.matrix[u][v]) for v in range(self.n)))
        return '\n'.join(lines)

class DirectedIncidenceMatrix:
    """
    Prosty graf skierowany reprezentowany jako macierz incydencji.
    -1 w wierszu ogona, +1 w wierszu głowy.
    """
    def __init__(self, source):
        if isinstance(source, DirectedAdjacencyList):
            n = source.n
            self.n = n
            edges = []
            for u in range(n):
                for v in source.adj[u]:
                    edges.append((u,v))
            m = len(edges)
            self.m = m
            self.matrix = [[0]*m for _ in range(n)]
            for idx,(u,v) in enumerate(edges):
                self.matrix[u][idx] = -1
                self.matrix[v][idx] =  1
        else:
            n = int(source)
            self.n = n
            self.m = 0
            self.matrix = []

    def __str__(self):
        if self.m == 0:
            return '\n'.join('' for _ in range(self.n))
        lines = []
        for u in range(self.n):
            lines.append(' '.join(str(self.matrix[u][e]) for e in range(self.m)))
        return '\n'.join(lines)


def generate_random_directed_graph_by_probability(n, p):
    """
    Generuje prosty losowy digraf G(n,p): każda możliwa krawędź u->v (u!=v)
    powstaje z prawdopodobieństwem p.
    Zwraca DirectedAdjacencyList.
    """
    if n < 1:
        raise ValueError("n musi być >= 1")
    if not (0 <= p <= 1):
        raise ValueError("p musi być w [0,1]")
    g = DirectedAdjacencyList(n)
    for u in range(n):
        for v in range(n):
            if u != v and random.random() < p:
                g.add_edge(u, v)
    return g

