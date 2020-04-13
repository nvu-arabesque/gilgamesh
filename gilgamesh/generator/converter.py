"""
    Convert from a type to another type
"""
import torch
import numpy as np
from gilgamesh.generator.graphs import AbstractGraph
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
        An example is given below:
        ```python
        edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)
        ```

        Args
        ---------
        G: generator.graphs.AbstractGraph
            a graph of type AbstractGraph or its subclasses
        
        Returns
        ---------
        Data: torch_geometric.data.Data
            Data of type torch_geometric
        """
        # 
        G.pygeom_prepare()
        # Get the appropriate feature
        num_nodes, num_node_features, node_feature = G.get_node_features()
        _, num_dim, node_pos = G.get_node_position()
        edge_list = G.get_edge_list()
        num_edge, num_edge_features, edge_attr = G.get_edge_attribute()
        label = G.get_label()
        # Some mid step
        # Convert edge list from list of edges (from, to) to list of from and list of to
        edge_list_tensor = torch.tensor(np.array(edge_list, dtype=np.int32),
            dtype=torch.long).t().contiguous()
        # Conversion other features to tensor
        node_feature_tensor = None
        if node_feature is not None and num_node_features > 0:
            node_feature_tensor = torch.tensor(np.array(node_feature).reshape(
            num_nodes, num_node_features), dtype=torch.float)
        
        edge_attribute_tensor = None
        if edge_attr is not None and num_edge_features > 0:
            edge_attribute_tensor = torch.tensor(np.array(edge_attr).reshape(
            num_edge, num_edge_features))

        node_pos_tensor = None
        if node_pos is not None and num_dim > 0:
            node_pos_tensor = torch.tensor(np.array(node_pos).reshape(
                num_nodes, num_dim))

        # Create Data object
        return Data(x=node_feature_tensor, edge_index = edge_list_tensor,
            edge_attr = edge_attribute_tensor, y=label, pos=node_pos_tensor)
        

