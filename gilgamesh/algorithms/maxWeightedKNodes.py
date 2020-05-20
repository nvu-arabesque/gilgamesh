import os, sys, random, math, itertools, heapq
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from functools import reduce
sys.path.append('../..')

import gilgamesh
from gilgamesh.generator.graphs import CompleteGraph



# CREDITS TO @LetterRip
def find_all_cycles(G):
    graph = [[edge[0], edge[1]] for edge in G.to_directed().edges]
    cycles = []
    def findNewCycles(path):
        start_node = path[0]
        next_node= None
        sub = []

        #visit each edge and each node of each edge
        for edge in graph:
            node1, node2 = edge
            if start_node in edge:
                if node1 == start_node:
                    next_node = node2
                else:
                    next_node = node1
                    if not visited(next_node, path):
                        # neighbor node not on path yet
                        sub = [next_node]
                        sub.extend(path)
                        # explore extended path
                        findNewCycles(sub);
                    elif len(path) > 2  and next_node == path[-1]:
                        # cycle found
                        p = rotate_to_smallest(path);
                        inv = invert(p)
                        if isNew(p) and isNew(inv):
                            cycles.append(p)

    def invert(path):
        return rotate_to_smallest(path[::-1])

    #  rotate cycle path such that it begins with the smallest node
    def rotate_to_smallest(path):
        n = path.index(min(path))
        return path[n:]+path[:n]

    def isNew(path):
        return not path in cycles

    def visited(node, path):
        return node in path

    for edge in graph:
        for node in edge:
            findNewCycles([node])
    return cycles

def makeGraph(size):
    G = CompleteGraph(size).get_graph()
    weightedEdges = {
                        (edge[0], edge[1]) : { 'weight': random.uniform(0,1) }
                        for edge in G.edges
                    }
    nx.set_edge_attributes(G, weightedEdges)
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
    groupsWithSumOfEdges = []
    for kGroup in kGroups:
        nodeSum = {}
        nodeSum["Nodes"] = kGroup
        nodeSum["Sum"] = (G.subgraph(kGroup)).size("weight")
        groupsWithSumOfEdges.append(nodeSum)

    maxGroup = reduce((lambda x, y: x if x["Sum"] < y["Sum"] else y), groupsWithSumOfEdges)
    return (maxGroup, groupsWithSumOfEdges)

def sortedEdges(k, G):
    finished = False
    subGraph = nx.Graph()
    edgeQueue = []
    for edge in G.edges.data("weight"):
        heapq.heappush(edgeQueue, (edge[2], edge[0], edge[1]))
    cycles = []
    while(not finished and len(edgeQueue) != 0):
        next_edge = heapq.heappop(edgeQueue)
        subGraph.add_edge(next_edge[2], next_edge[1], weight=next_edge[0])
        try:
            cycles = find_all_cycles(subGraph)
        except nx.exception.NetworkXNoCycle:
            cycles = []
        filter(lambda x: len(x) == k, cycles)
        finished = len(cycles) > 0
    allSubGraphs = [G.subgraph(cycle) for cycle in cycles]
    allSubGraphsWithSum = [(graph.nodes, graph.size("weight")) for graph in allSubGraphs]
    return min(allSubGraphsWithSum, key = lambda x: x[1])

def findMaxWeightedKNodes(k, completeGraph):
    bruteStartTime = datetime.now()
    brute = bruteForce(k, completeGraph)[0]
    bruteElapsedTime = (datetime.now() - bruteStartTime).total_seconds()
    sortStartTime = datetime.now()
    sorted = sortedEdges(k, completeGraph)
    sortedElapsedTime = (datetime.now() - sortStartTime).total_seconds()
    print("BruteForce result: \n\t Nodes: %s \n\t Sum: %s\n\t Elapsed time: %ld" % (brute['Nodes'], brute['Sum'], bruteElaspedTime))
    print("SortedEdge result: \n\t Nodes: %s \n\t Sum: %s\n\t Elapsed time: %ld" % (sorted[0], sorted[1], sortedElapsedTime))

def generateGraphWithAnswer(size, k):
    assert k <= size
    findMaxWeightedKNodes(k, makeGraph(size))



def main():
    size = 50
    k = 20
    print("Graph size: %d, K-size: %d \n\n *************************************" % (size, k))
    generateGraphWithAnswer(size, k)

main()
