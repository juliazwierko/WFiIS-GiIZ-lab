import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from MyGraph import  *
from DirectedMyGraph import *

def Draw(graph: Graph, filename: str = "graph.png", legend_title: str = "Graph", output_dir: str = "outputs/04", with_weights: bool = False, mst: list[tuple[int, int, int]] = None ) -> None:
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
          
        if mst:
            mst_edges = [(u, v) for u, v, _ in mst]
            nx.draw_networkx_edges(G, pos, edgelist=mst_edges, edge_color="red", width=2)
            
        plt.legend([legend_title], loc="upper right", fontsize=12)

        plt.axis("off") 
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        # print(f"Graf zapisany jako {filename}")
    except Exception as e:
        print(f"Błąd podczas rysowania grafu: {e}")


def Draw_Flow_Network(graph: FlowNetwork, legend_title: str, filename: str = 'flow_network.png', output_dir: str = "outputs/05"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    
    try:
        G = nx.DiGraph()
        weighted_edges = [(u, v, w) for (u, v), w in graph.weighted_edges.items()]
        G.add_weighted_edges_from(weighted_edges)

        layers = graph.internal_layers
        pos = {}
        for layer_idx, layer in enumerate(layers):
            y_step = 1.0
            #y_step = 1.0 / (len(layer) + 1)
            for i, node in enumerate(layer):
                pos[node] = (layer_idx, (i + 1) * y_step)

        plt.figure(figsize=(10, 5))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_size = 10, arrows=True, connectionstyle="arc3,rad=0.1")
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, connectionstyle="arc3,rad=0.1")
        plt.legend([legend_title], loc="upper center", fontsize=12)
        plt.axis('off')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graf zapisany jako {filepath}")
        
    except Exception as e:
        print(f"Błąd podczas rysowania grafu: {e}")




    