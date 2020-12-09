"""
    Abstract class for problem set,
    Problem will inherit from here
"""
import os, sys, math, random, matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod

class AbstractGraph(ABC):
    """ Skeleton for problems """

    def __init__(self, size):
        self._graph = None
        self._adj_matrix = None
        self._size = None

        self._node_features = None
        self._node_positions = None
        self._edge_list = None
        self._edge_features = None
        self._label = None

    @abstractmethod
    def is_graph_valid(self):
        pass

    @abstractmethod
    def populate_graph(self, size):
        pass

    def get_graph(self):
        if self._graph is None:
            print("Graph has not been populated")
            return None
        return self._graph

    def get_adj_matrix(self):
        if self._adj_matrix is None:
            print("Graph has not been populated")
            return None
        return self._adj_matrix

    def draw(self):
        if self._graph is None:
            print("Graph has not been populated")
        else:
            nx.draw(self._graph)

    def breakdown(self):
        """ """
        E = list(nx.generate_edgelist(self._graph,
            delimiter=','))
        self._edge_list = [x.split(',')[:-1] for x in E]
        self._edge_features = [x.split(',')[-1] for x in E]

    def get_node_features(self):
        """
        
        Returns
        ----------
        tuple: (int, int, list)
            num_nodes, num_node_features, node_feature    
        """
        num_nodes = self._size
        num_node_features = 0 if self._node_features is None else len(self._node_features[0])
        node_feature = self._node_features
        return (num_nodes, num_node_features, node_feature)

    def get_edge_list(self):
        """ """
        return self._edge_list

    def get_edge_attribute(self):
        """
        Returns
        ---------
        tuple: (int, int, list)
            num_edge, num_edge_features, edge_attr
        """
        num_edge = len(self._edge_list)
        num_edge_features = 0 if self._edge_features is None else len(self._edge_features[0])
        edge_attr = self._edge_features
        return (num_edge, num_edge_features, edge_attr)

    def get_node_position(self):
        """
        Returns
        ----------
        tuple: (int, int, list)
            num_nodes, num_dim, node_pos
        """
        num_nodes = self._size
        num_dim = 0 if self._node_positions is None else len(self._node_positions[0])
        node_pos = self._node_positions
        return (num_nodes, num_dim, node_pos)

    def get_label(self):
        """

        Returns
        --------
        label: list
        """
        return self._label

class CompleteGraph(AbstractGraph):        
    @staticmethod
    def is_complete(G):
        n = G.number_of_nodes()
        for node in G.nodes():
            if len(G[node]) < (n - 1):
                return False
        return True

class LineGraph(AbstractGraph):
    @staticmethod
    def is_line(G):
        visited = []
        for node in nx.Graph.nodes(G):
            if node in visited:
                return False
            visited.append(node)
        return len(visited) == nx.Graph.number_of_nodes(G)


class CycleGraph(AbstractGraph):
    @staticmethod
    def is_cycle(G):
        return nx.is_directed_acyclic_graph(G.to_directed())
