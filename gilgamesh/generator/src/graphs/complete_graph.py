import os
import sys
import networkx as nx
import numpy as np
from abstract_graph import AbstractGraph
from abc import ABC, abstractmethod

class CompleteGraph(AbstractGraph):
    def __init__(self, train_size, test_size):
        AbstractGraph.__init__(self, train_size, test_size)

    def populate_graph(self):
        try:
            graph = nx.complete_graph(size)
        except Exception:
            print("Nx could not generate graph")

    @staticmethod
    def is_graph_valid(G):
        n = nx.Graph.number_of_nodes(G)
        for node in nx.Graph.nodes(G):
            if  len(nx.Graph.neighbors(node)) != (n - 1):
                return False
        return True
