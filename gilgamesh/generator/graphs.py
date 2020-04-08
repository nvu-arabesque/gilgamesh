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

    def __init__(self):
        self._graph = None
        self._adj_matrix = None
        self._size = None

    @abstractmethod
    def is_graph_valid(G):
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

class CompleteGraph(AbstractGraph):

    def __init__(self):
        AbstractGraph.__init__(self)

    def populate_graph(self, size):
        self._size = size
        self._graph = nx.complete_graph(size)
        self._adj_matrix = nx.to_numpy_matrix(self._graph, dtype=np.int64)


    @staticmethod
    def is_graph_valid(G):
        n = G.number_of_nodes()
        for node in G.nodes():
            if len(G[node]) < (n - 1):
                return False
        return True

class LineGraph(AbstractGraph):
    def __init__(self):
        AbstractGraph.__init__(self)

    def populate_graph(self, size):
        self._size = size
        indices = list(range(self._size))
        self._graph = nx.Graph()
        for i in range(self._size - 1):
            self._graph.add_edge(indices[i], indices[i+1])
        self._adj_matrix = nx.to_numpy_matrix(self._graph, dtype=np.int64)

    @staticmethod
    def is_graph_valid(G):
        visited = []
        for node in nx.Graph.nodes(G):
            if node in visited:
                return False
            visited.append(node)
        return len(visited) == nx.Graph.number_of_nodes(G)


class CycleGraph(AbstractGraph):
    def __init__(self):
        AbstractGraph.__init__(self)

    def populate_graph(self, size):
        self._size = size
        self._graph = nx.cycle_graph(self._size)
        self._adj_matrix = nx.to_numpy_matrix(self._graph, dtype=np.int64)


    @staticmethod
    def is_graph_valid(G):
        return nx.is_directed_acyclic_graph(G.to_directed())
