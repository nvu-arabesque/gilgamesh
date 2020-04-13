"""
    Convert from a type to another type
"""
import torch
from src.graphs import AbstractGraph
from torch_geometric.data import Data

class Converter:

    @staticmethod
    def to_pytorch_geometric(G: AbstractGraph) -> Data:
        """ Convert a graph objecet to pytorch geometric object 
        
        The geometric format by default is:

            - data.x: Node feature matrix with shape [num_nodes, num_node_features]
            - data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
            - data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            - data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]
            - data.pos: Node position matrix with shape [num_nodes, num_dimensions]
        
        Args
        ---------
        G: generator.graphs.AbstractGraph
            a graph of type AbstractGraph or its subclasses
        
        Returns
        ---------
        Data: torch_geometric.data.Data
            Data of type torch_geometric
        """
        x = G.V
        edge_list = G.E
        edge_attr = G.edge_attr()
        
        

