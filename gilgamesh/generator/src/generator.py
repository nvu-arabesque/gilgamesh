import os
import sys
import numpy as np
import networkx as nx
import glob
from graphs import *
from enum import Enum

class GraphTypes(Enum):
    COMPLETE = CompleteGraph
    LINE = LineGraph
    CYCLE = CycleGraph


def generate_graph(graph_type, graph_size, graph_sparsity, train_size, test_size):
    data = {}
    train_set = []
    test_set = []
    assert len(graph_type) == len(graph_size) and len(graph_size) == len(graph_sparsity)
    for i in range(len(graph_type)):
        G_type = graph_type[i]
        if G_type in GraphTypes.__members__.keys():
            G = GraphTypes[G_type].value(graph_size[i])
            G.populate_graph()
            train_set.append(G)
    for G in train_set:
        print(G.get_graph())

type = ['COMPLETE', 'LINE', 'CYCLE']
size = [10, 10, 15]

generate_graph(type, size, [0.5, 0.5, 0], 1, 1)
