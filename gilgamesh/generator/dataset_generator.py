import os, sys
import numpy as np
import logging
import math
from gilgamesh.generator.graph_generator import UndirectedGraphGenerator
from gilgamesh.generator.utils import chunkify


class GraphGenerator:
    G_TYPES = {
        "path": UndirectedGraphGenerator.generate_path,
        "cycle": UndirectedGraphGenerator.generate_cycle,
        "complete": UndirectedGraphGenerator.generate_complete,
        "kd": UndirectedGraphGenerator.generate_random_regular,
        "k_circulant": UndirectedGraphGenerator.generate_k_circulant,
        "gnp": UndirectedGraphGenerator.generate_gnp,
    }

    @staticmethod
    def generate_undirected_graph(size: int, g_type: str, **kwargs):
        assert g_type in GraphGenerator.G_TYPES, f"Graph Type not available: {g_type}"
        return GraphGenerator.G_TYPES[g_type](size, **kwargs)

    def write_data(self, path):
        ts = time.gmtime()
        filename = time.strftime("%Y-%m-%d_%H:%M:%S", ts) + ".json"
        outdir = os.path.join("out/", path)
        outpath = os.path.join(outdir, filename)
        if not os.path.exists(outdir) or os.path.isfile(outdir):
            os.makedirs(outdir)
        with open(outpath, "w") as file:
            file.write(json.dumps(self.data))

        print("Success, file written %s " % outpath)


class IsPathDataset:
    """Generate a dataset for problem of identifying
    whether a graph is a path.
    """

    @staticmethod
    def generate(
        n_samples: int,
        train_test_ratio: float = 0.8,
        label_ratio: float = 0.5,
        graph_types: str = "mixture",
    ):
        """Returns a dataset consists of train and set,
        determined by the input n_samples and ratio.

        Types of graph varied.

        Returns
        ---------
        dataset: Dict
            Dataset as a dictionary contains two keys: train and set
            respectively.
        """
        """Generate a dataset for classifying whether a graph is a path.

        Args
        -------
        n: int
            number of test
        label_ratio: float
            pos/neg ratio

        Returns
        ---------
        Gs: List of graphs
            list of graphs contain n number of samples.
        """
        assert label_ratio < 1 and label_ratio > 0, "label_ratio should be \in (0,1)"
        n_pos = math.floor(n * label_ratio)
        n_neg = math.floor(n * (1 - label_ratio))
        # Generate positive labels
        # note we add 2 to ensure the length is at least 2
        pos_g_size = np.random.randint(max_size, size=n_pos) + 2
        pos = [UndirectedGraphGenerator.generate_path(x) for x in pos_g_size]
        # Generate negative labels, non-path
        neg_types = ["cycle", "complete"]
        neg_g_size = np.random.randint(max_size, size=n_neg)
        neg_g_samples = chunkify(neg_g_size, len(neg_types))
        neg = [
            GraphGenerator.generate_undirected_graph(g_type=_type, size=_size)
            for _type, _size_ in zip(neg_types, neg_g_samples)
            for _size in _size_
        ]
        # collect statistics of dataset
        return {"True": pos, "False": neg}


# class IsConnectedDataset:
#     """ Generate datasets for graphs which are:
#         - disconnected
#         - connected

#         Graph models used for generation are:
#         - paths
#         - cycle
#         - Gnp
#         - r-regular graph
#         - Complete Graph
#         - Perfectly Balanced Tree

#         For disconnected graph:
#         - multiple paths
#         - multiple cycles
#         - Gnp: p < 1/sqrt(n)
#         - r-regular
#         - multiple complete graphs
#         - multiple perfectly balanced tree
#         - perhaps should add some scaled-free network
#     """
#     def __init__(self, logger: object = None):
#         self._logger = logger if logger is not None else\
#             logging.getLogger(__name__)

#     def _generate_connected(self, sizes: int):
#         """
#             Generate connected graphs.

#             Inputs is a 2d array which contains the parameter of the
#                 graph in each

#             Args
#             -------
#             sizes: 2D arrays
#                 each


#             TODO: take a dictionary for specifying how many graphs of different
#             type
#         """
#         # generate path
#         # generate gnp
#         # generate r-regular
#         # generate complete graphs
#         # generate perfect trees
#         # generate scaled-free network


# class IsCompleteDataset:

# class IsRegularDataset:

# class IsTreeDataset:

# class IsPlanarDataset:

# class IsBipartiteDataset:

# class IsPartiteDataset: