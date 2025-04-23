import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from MyGraph import  *

def Draw(graph: Graph, filename: str = "graph.png", legend_title: str = "Graph", output_dir: str = "outputs/03", with_weights: bool = False) -> None:
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
                    
        # checking whether edge weights are set properly            
        #print(graph.weights)
        #print(nx.get_edge_attributes(G, 'weight'))
        
            nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), font_size=8)
            
        plt.legend([legend_title], loc="upper right", fontsize=12)

        plt.axis("off") 
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Graf zapisany jako {filename}")
    except Exception as e:
        print(f"Błąd podczas rysowania grafu: {e}")
