import os
import sys
from enum import Enum
from complete_graph import CompleteGraph
from line_graph import LineGraph
from abstract_graph import AbstractGraph

class GraphProperties(Enum):
    CONNECTED = is_graph_connected
    COMPLETE = LineGraph
