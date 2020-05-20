import os, sys, random, math, itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import timeit

sys.path.append('../..')

import gilgamesh
from gilgamesh.generator.graphs import CompleteGraph


seed = 10

def makeGraph(size):
    G = CompleteGraph(size).get_graph()
    weightedEdges = {
                        (edge[0], edge[1]) : { 'weight': random.uniform(0,1) }
                        for edge in G.edges
                    }
    nx.set_edge_attributes(G, weightedEdges)
    drawGraph(G)
    return G

def drawGraph(G):
    """
        G = networkx Graph
    """
    precision = 2
    pos = nx.spring_layout(G)
    nx.draw(G,pos,edge_color='black',width=1,linewidths=1,
            node_size=500,node_color='pink',alpha=0.9,
            labels={node:node for node in G.nodes()})
    labels = {(edge[0], edge[1]) : round(edge[2]['weight'], precision) for edge in G.edges.data()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def bruteForce(k, G):
    """
        Args: 
            k - size of components
            G - networkX graph

    """

    kGroups = list(itertools.combinations(list(G.nodes), k))
    for kGroup in kGroups:
        subgraphG = G.subgraph(kGroup)
    groupsWithSumOfEdges = [
                            dict({
                                "Nodes" : kGroup,
                                "Sum" : G.subgraph(kGroup).size() 
                            }) for kGroup in kGroups ]    
    return groupsWithSumOfEdges

def findMaxWeightedKNodes(k, completeGraph):
    bruteForce(k, completeGraph)

def generateGraphWithAnswer(size, k):
    assert k <= size
    findMaxWeightedKNodes(k, makeGraph(size))

generateGraphWithAnswer(10, 4)