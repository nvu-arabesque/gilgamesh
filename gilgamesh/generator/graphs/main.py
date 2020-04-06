import os
import sys
import random
import abstract_graph
import line_graph
import complete_graph


graphs = []

for i in range(10):
    x = random.random()
    G = null
    if x > 0.5:
        G = CompleteGraph(10, 10)
    else:
        G = LineGraph(10,10)
