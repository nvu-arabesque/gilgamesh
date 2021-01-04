import os, sys, glob, matplotlib, random, math, time, json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from generator.graph_generator import GraphGenerator
from enum import Enum

import logging

class GraphsError(Exception):
    pass

class InputError(GraphsError):
    """Exception raised when user makes input error

    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class GraphTypes(Enum):
    COMPLETE = CompleteGraph
    LINE = LineGraph
    CYCLE = CycleGraph



class Dataset():
    """Each instance of Dataset will hold exactly 1 dictionary containing:

        (1) A set of training graphs
        (2) A set of test graphs

        There are only two public functions which are self_explanatory:
            (1) generate_dataset()  (2) get_dataset()


        NOTE: At the moment all ouptut will be true

    """

    def __init__(self):
        self.train_set = []
        self.test_set = []


    def check_inputs(self, types, sizes, sparsities):
        """
            Checks that the user input is valid
        """
        if len(types) != len(sizes) or len(sizes) != len(sparsities):
            raise InputError([types, sizes, sparsities], "Input size mismatched!")

        if all(isinstance(x, (int, float)) for x in sparsities):
            if sum(sparsities) != 100:
                raise InputError(sparsities, "Total density must equals to 100(%)")
        else:
            raise InputError(sparsities, "Densities of graphs must be ints adding up to 100 or floats added up to 1")


    def create_graph(self, G_type, G_max_size, is_graph_in_training_set):
        """
            Helper method to create a single graph that returns False if
            there is any error
        """
        if G_type in GraphTypes.__members__.keys():
            G = GraphTypes[G_type].value()
            G.populate_graph(random.randint(1, G_max_size))
            if is_graph_in_training_set:
                self.train_set.append(G)
            else:
                self.test_set.append(G)
            return True
        else:
            return False

    def generate_dataset(self, types, max_graph_sizes, sparsities, train_size, test_size):
        """
            Function responsible for generating a set of trainning and testing graphs

        """
        self.check_inputs(types, max_graph_sizes, sparsities)
        if not (isinstance(train_size, int)  and isinstance(test_size, int)):
            raise InputError([train_size, test_size], "Sizes must be integers")
        if not (train_size >= 0 and test_size >= 0):
            raise InputError([train_size, test_size], "Sizes must be positive")

        train_no = [round(n * train_size / 100) for n in sparsities]
        test_no = [round(n * test_size / 100) for n in sparsities]
        graphs_data = zip(types, max_graph_sizes, train_no, test_no)


        for (G_type, G_max_size, train_samples, test_samples) in graphs_data:
            for i in range(train_samples):
                self.create_graph(G_type, G_max_size, True)
            for j in range(test_samples):
                self.create_graph(G_type, G_max_size, False)



    def get_dataset_as_graphs(self):
        """
            Returns a dictionary containing a training and a testing set
        """
        data = {}
        data["train"] = self.train_set
        data["test"] = self.test_set
        return data


    def get_dataset_as_matrices(self):
        data = {}
        data["train"] = [G.get_adj_matrix().tolist() for G in self.train_set]
        data["test"] = [G.get_adj_matrix().tolist() for G in self.test_set]
        return data

    def pretty_print_model(self):
        for G in self.train_set + self.test_set:
            G.draw()
            plt.show()


class GraphGenerator():

    @classmethod
    def check_inputs(cls, properties, train_set_size, test_set_size):
        if not all(p in GraphTypes.__members__.keys() for p in properties):
            raise InputError(properties, "Graph property not suppoted")
        if train_set_size < 0 or train_set_size < 0:
            raise InputError((train_set_size, test_set_size), "Sample size must be positive integers")

    def __init__(self, properties, train_set_size, test_set_size):
        """
            INPUT:

            (1) Properties
                A dictionary containing:
                - Characteristics of each graph for classification
                - Maximum size for each type of graph

            (2) Size of training set
            (3) Size of test set

            --------------------------------------------------------------
            SAMPLE:

            GraphGenerator([COMPLETE', 'LINE', 'CYCLE'], 10, 10)

            Only graphs which are 'Complete', 'line' and 'cycle' will have an
            output of TRUE

        """
        GraphGenerator.check_inputs(properties, train_set_size, test_set_size)
        self.properties = properties
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.data = None


    def compute_result(self, dataset):
        """
            Given a dataset of arbitrary types of graphs, generate a dictionary
            which contains the graphs with its expected output

            i.e.: True <=> the graph has all provided properties


        """
        train_set = []
        test_set = []
        for G in dataset.get_dataset_as_graphs()["train"]:
            nxG = G.get_graph()
            validity = all(GraphTypes[GType].value().is_graph_valid(nxG) for GType in self.properties)
            temp = {}
            temp["input"] = G.get_adj_matrix().tolist()
            temp["output"] = validity
            train_set.append(temp)

        for G in dataset.get_dataset_as_graphs()["test"]:
            nxG = G.get_graph()
            validity = all(GraphTypes[GType].value().is_graph_valid(nxG) for GType in self.properties)
            temp = {}
            temp["input"] = G.get_adj_matrix().tolist()
            temp["output"] = validity
            test_set.append(temp)

        data = {}
        data["train"] = train_set
        data["test"] = test_set
        print("train size:" + str(len(train_set)))
        print("test size:" + str(len(test_set)))
        self.data = data


    def generate(self):
        """
        Public method for generating a dataset for the model

        """
        random.seed(1)
        dataset = Dataset()
        sparsities = []
        sizes = []
        max_density = 100
        for i in range(1, len(GraphTypes.__members__.keys())):
            cur = random.randint(0, max_density)
            max_density -= cur
            sparsities.append(cur)
            sizes.append(random.randint(1, 20))

        sparsities.append(max_density)
        sizes.append(random.randint(1, 20))
        dataset.generate_dataset(GraphTypes.__members__.keys(), sizes , sparsities, self.train_set_size, self.test_set_size)
        self.compute_result(dataset)


    def get_data(self):
        return self.data


    def write_data(self, path):
        ts = time.gmtime()
        filename = time.strftime("%Y-%m-%d_%H:%M:%S", ts) + ".json"
        outdir = os.path.join("out/", path)
        outpath = os.path.join(outdir, filename)
        if not os.path.exists(outdir) or os.path.isfile(outdir):
            os.makedirs(outdir)
        with open(outpath, "w") as file:
            file.write(json.dumps(self.data))

        print('Success, file written %s ' % outpath)

def DatasetTest():
    type = ['COMPLETE', 'LINE', 'CYCLE']
    size = [10, 10, 15]
    sparsity = [0.25, 0.25, 0.5]
    GraphTestOne = Dataset()
    GraphTestOne.generate_dataset(type, size, sparsity, 10, 10)


def GeneratorTest():
    gen = GraphGenerator(['COMPLETE'], 10, 10)
    gen.generate()
    gen.write_data("1")

# DatasetTest()
GeneratorTest()

class Dataset:
    def __init__(self, n, r, types):
        """
            n: number of graphs
            r: ratio
            size: [x, y]
            types: ['cycle', 'tree', etc.]
        """

    def generate_balanced():
        """ Go through from range(n)/2, roll two dice pick for type and size,
        generate satisfied, same for 
        """


class IsConnectedDataset:
    """ Generate datasets for graphs which are:
        - disconnected
        - connected

        Graph models used for generation are:
        - paths
        - cycle
        - Gnp
        - r-regular graph
        - Complete Graph
        - Perfectly Balanced Tree

        For disconnected graph:
        - multiple paths
        - multiple cycles
        - Gnp: p < 1/sqrt(n)
        - r-regular
        - multiple complete graphs
        - multiple perfectly balanced tree
        - perhaps should add some scaled-free network
    """
    def __init__(self, logger: object = None):
        self._logger = logger if logger is not None else\
            logging.getLogger(__name__)

    def _generate_connected(self, sizes: int):
        """
            Generate connected graphs.

            Inputs is a 2d array which contains the parameter of the
                graph in each 

            Args
            -------
            sizes: 2D arrays
                each
                

            TODO: take a dictionary for specifying how many graphs of different
            type
        """
        # generate path
        # generate gnp
        # generate r-regular
        # generate complete graphs
        # generate perfect trees
        # generate scaled-free network


class IsCompleteDataset:

class IsRegularDataset:

class IsTreeDataset:

class IsPlanarDataset:

class IsBipartiteDataset:

class IsPartiteDataset: