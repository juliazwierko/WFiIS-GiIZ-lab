import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from MyGraph import  *

def Draw(graph: Graph, filename: str = "graph.png", legend_title: str = "Graph", output_dir: str = "outputs/03", with_weights: bool = False, mst: list[tuple[int, int, int]] = None ) -> None:
    """
    Draws graph with optional MST and egde weights.

    Args:
        graph (Graph): The input graph
        filename (str): Name of the output file
        legend_title (str): Title of the graph
        output_dir (str): Output directory
        with_weights (bool): Whether to display edge weights
        mst (list[tuple[int, int, int]]): List of MST edges in the format (u, v, w)
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    
    if graph.type in {GraphRepresentationType.AdjacencyList, GraphRepresentationType.IncidenceMatrix}:
        graph = graph.to_AM()

    adjacency_matrix = np.array(graph.data)
    try:
        G = nx.from_numpy_array(adjacency_matrix)
        plt.figure(figsize=(6, 6))

        n = len(G.nodes) 
        r = 1 
        x0, y0 = 0, 0  
        
        alpha = 2 * np.pi / n
        pos = {}
        
        for i, node in enumerate(G.nodes):
            xi = x0 + r * np.cos(i * alpha)
            yi = y0 + r * np.sin(i * alpha)
            pos[node] = (xi, yi)
        
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", font_weight="bold", font_size=10)
        
        
        if with_weights:
            for (i, j), weight in graph.weights.items():
                if G.has_edge(i, j):
                    G[i][j]['weight'] = weight
            nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), font_size=8, label_pos=0.55)
            
        # checking whether edge weights are set properly            
        # print(f'graph.weights : {graph.weights}')            # wagi krawędzi są zawsze posortowane według pierwszego weirzchołka a pary zawsze sa od mniejszego do większego
        # print(nx.get_edge_attributes(G, 'weight'))
          
        if mst:
            mst_edges = [(u, v) for u, v, _ in mst]
            nx.draw_networkx_edges(G, pos, edgelist=mst_edges, edge_color="red", width=2)
            
        plt.legend([legend_title], loc="upper right", fontsize=12)

        plt.axis("off") 
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Graf zapisany jako {filename}")
    except Exception as e:
        print(f"Błąd podczas rysowania grafu: {e}")
