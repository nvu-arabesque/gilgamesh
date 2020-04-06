import os
import sys
import networkx as nx
import numpy as np
from abstract_graph import AbstractGraph
from abc import ABC, abstractmethod

class LineGraph(AbstractGraph):
    def __init__(self, size):
        AbstractGraph.__init__(self, train_size, test_size)

    def populate_graph(self):
        try:
            graph = nx.line_graph(size)
        except Exception:
            print("Networkx could not generate graph")

    @staticmethod
    def is_graph_valid(G):
        visited = []
        for node in nx.Graph.nodes(G):
            if node in visited:
                return False
            visited.append(node)
        return len(visited) == nx.Graph.number_of_nodes(G)
