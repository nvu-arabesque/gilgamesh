"""
    Abstract class for problem set,
    Problem will inherit from here
"""
import os, sys, math, random
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod

class AbstractGraph(ABC):
    """ Skeleton for problems """

    def __init__(self, size):
        self._size = size
        self._adj_matrix = None

    @abstractmethod
    def is_graph_valid(self):
        pass

    @abstractmethod
    def populate_graph(self):
        pass

    def get_graph(self):
        if self._adj_matrix is None:
            print("Graph has not been populated")
            return None
        return self._adj_matrix

class CompleteGraph(AbstractGraph):

    def __init__(self, size):
        AbstractGraph.__init__(self, size)

    def populate_graph(self):
        G = nx.complete_graph(self._size)
        self._adj_matrix = nx.to_numpy_matrix(G, dtype=np.int64)


    @staticmethod
    def is_graph_valid(G):
        n = nx.Graph.number_of_nodes(G)
        for node in nx.Graph.nodes(G):
            if  len(nx.Graph.neighbors(node)) < (n - 1):
                return False
        return True

class LineGraph(AbstractGraph):
    def __init__(self, size):
        AbstractGraph.__init__(self, size)

    def populate_graph(self):
        indices = list(range(0, self._size))
        random.shuffle(indices)
        G = nx.Graph()
        for i in range(self._size - 1):
            G.add_edge(indices[i], indices[i + 1])
        self._adj_matrix = nx.to_numpy_matrix(G, dtype=np.int64)

    @staticmethod
    def is_graph_valid(G):
        visited = []
        for node in nx.Graph.nodes(G):
            if node in visited:
                return False
            visited.append(node)
        return len(visited) == nx.Graph.number_of_nodes(G)


class CycleGraph(AbstractGraph):
    def __init__(self, size):
        AbstractGraph.__init__(self, size)

    def populate_graph(self):
        G = nx.cycle_graph(self._size)
        self._adj_matrix = nx.to_numpy_matrix(G, dtype=np.int64)


    @staticmethod
    def is_graph_valid(G):
        return nx.is_directed_acyclic_graph(G.to_directed())
