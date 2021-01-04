"""
    Light wrapper around networkx to generate graph of different kind.
    TODO:
    - path
    - cycle
    - Kn (Complete Graph of Order n)
    - K_{n,m} complete bipartite
"""
import os, sys, glob, matplotlib, random, math, time, json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from typing import List
from enum import Enum


class OutputType(Enum):
    """ TODO: to rotate between different representation """

    Matrix = "Matrix"
    Adj = "Adjacency"


class UndirectedGraphGenerator:
    # ================================================================================= #
    # Generate multiple graphs
    # ================================================================================= #
    @staticmethod
    def generate_multiple_graphs(kind: str, sizes: List[int]):
        """ Generate a list of paths.

            Path_i's is of order sizes_i.

            Args
            ------
            sizes: List of int
                where sizes[i] specifies the size of path_i.

            Returns
            --------
            paths: List of paths
        """
        assert kind in ['path', 'cycle', 'Kn', 'perfect_tree', 'Gnp', 'r-regular']
        if kind in ['path', 'cycle', 'Kn', 'perfect_tree']:
            return UndirectedGraphGenerator._generate_multiple_simple_parametered_model(sizes)
        else:
            raise NotImplementedError
        return [UndirectedGraphGenerator.generate_path(x) for x in sizes]

    @staticmethod
    def _generate_multiple_simple_parametered_model(kind: str, sizes: List[int]):
        """
        """
        assert all(x > 0 for x in sizes)
        generate_f = None
        if kind == 'path':
            generate_f = UndirectedGraphGenerator.generate_path
        elif kind == 'cycle':
            generate_f = UndirectedGraphGenerator.generate_cycle
        elif kind == 'Kn':
            generate_f = UndirectedGraphGenerator.generate_complete
        elif kind == 'perfect_tree':
            generate_f = UndirectedGraphGenerator.generate_perfect_balanced_tree
        else:
            raise Exception('Unexpected Graph Model Generation! Got: {}'.format(kind))
        return [generate_f(x) for x in sizes]

    # ================================================================================= #
    # Generate single graph
    # ================================================================================= #
    @staticmethod
    def generate_path(n: int):
        """ Generate a path of order n.
        """
        indices = list(range(n))
        G = nx.Graph()
        for i in range(n - 1):
            G.add_edge(indices[i], indices[i + 1])
        return nx.to_numpy_matrix(G, dtype=np.int64)

    @staticmethod
    def generate_cycle(n: int):
        """ Generate a cycle of order n.
        """
        return nx.to_numpy_matrix(nx.cycle_graph(n), dtype=np.int64)

    @staticmethod
    def generate_complete(n: int):
        """ Generate K_n """
        return nx.to_numpy_matrix(nx.complete_graph(n), dtype=np.int64)

    @staticmethod
    def generate_complete_bipartite(n: int, m: int):
        """ Generate complete bipartite graph.

            Args
            -----
            n: int
                number of vertices in the first set
            m: int
                number of vertices in the second set
        """
        return nx.to_numpy_matrix(nx.complete_bipartite_graph(n, m), dtype=np.int64)

    @staticmethod
    def generate_barbell(n: int, m: int):
        """ Generate barbell graph.

            Args
            -----
            n: int
                number of vertices in the first Kn
            m: int
                number of vertices in the second Kn
        """
        return nx.to_numpy_matrix(nx.barbell(n, m), dtype=np.int64)

    @staticmethod
    def generate_perfect_balanced_tree(t: int, h: int):
        """ Generate a perfectly balanced t-nary tree of size h.    
        """
        return nx.to_numpy_matrix(nx.balanced_tree(t, h), dtype=np.int64)

    @staticmethod
    def generate_k_circulant(n: int, k: int):
        """ Generate a k circulant graph of order n.
        """
        return nx.to_numpy_matrix(
            nx.generators.classic.circulant_graph(n, list(range(1, k + 1))),
            dtype=np.int64,
        )

    @staticmethod
    def generate_random_regular(n: int, d: int):
        """ Generate random regular graph of degree d. Using Kim & Vu method.
            Note that n * d must be even.
        """
        return nx.to_numpy_matrix(
            nx.generators.random_graphs.random_regular_graph(n, d),
        )

