"""
    Generate graph of different kind.
    TODO:
    - path
    - cycle
    - Kn (Complete Graph of Order n)
    - 
"""
import os, sys, glob, matplotlib, random, math, time, json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Int
from enum import Enum

class OutputType(Enum):
    Matrix = 'Matrix'
    Adj = 'Adjacency'

class GraphGenerator:
    @staticmethod
    def generate_path(n: Int):
        """ Generate a path of order n.
        """
        indices = list(range(n))
        G = nx.Graph()
        for i in range(n - 1):
            G.add_edge(indices[i], indices[i+1])
        return nx.to_numpy_matrix(G, dtype=np.int64)

    @staticmethod
    def generate_cycle(n: Int):
        """ Generate a cycle of order n.
        """
        return nx.to_numpy_matrix(nx.cycle_graph(n), dtype=np.int64)

    @staticmethod
    def generate_complete(n: Int):
        """ Generate K_n """
        return nx.to_numpy_matrix(nx.complete_graph(n), dtype=np.int64)