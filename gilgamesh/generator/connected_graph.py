import numpy as np
import os
import networkx as nx
import math

def generate_connected_graph(type, n, p):
    if type == 4 :
        return connected_graph_types[4](n, p)
    elif type in connected_graph_types.keys():
        return connected_graph_types[type](n)
    else:
        print("Invalid type!")

def complete_graph(n):
    return nx.complete_graph(n)

def cycle_graph(n):
    return nx.cycle_graph(n)

def line_graph(n):
    return nx.line_graph(complete_graph(int(math.floor(math.sqrt(n)))))

def random_graph(n, p):
    return nx.gnp_random_graph(n, p)


connected_graph_types = dict({ 1 : line_graph,
                               2 : cycle_graph,
                               3 : complete_graph,
                               4 : random_graph})
