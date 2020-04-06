"""
    Abstract class for problem set,
    Problem will inherit from here
"""
import os
import sys
from abc import ABC, abstractmethod

class AbstractGraph(ABC):
    """ Skeleton for problems """

    def __init__(self, size):
        self.graph = None

    @abstractmethod
    def is_graph_valid(graph):
        pass

    @abstractmethod
    def populate_graph(self):
        pass

    def get_graph(self):
        if graph == None:
            print("Graph has not been populated")
            return None
        return graph

    # @abstractmethod
    # def populate_train(self):
    #     pass
    #
    # @abstractmethod
    # def populate_test(self):
    #     pass
    #
    # @abstractmethod
    # def populate_train_output(self):
    #     pass
    #
    # @abstractmethod
    # def populate_test_output(self):
    #     pass
    #
    # def populate(self):
    #     populate_train()
    #     populate_test()
    #     populate_train_output()
    #     populate_test_output()
    #
    # def get_train(self):
    #     return graphs_info["train"]
    #
    # def get_test(self):
    #     return graphs_info["test"]
    #
    # def get_train_output(self):
    #     return graphs_info["train_output"]
    #
    # def get_test_output(self):
    #     return graphs_info["test_output"]
    #
    # def get_data_pairs(self, is_train_data):
    #     graph_list = []
    #     if is_train_data:
    #         src_data = graphs_info["train"]
    #         src_result = graphs_info["train_data"]
    #     else:
    #         src_data = graphs_info["test"]
    #         src_results = graphs_info["test_data"]
    #
    #     for i in range(len(source)):
    #         g_data = {}
    #         g_data["input"] = graphs_info[src_data][i]
    #         g_data["output"] = graphs_info[src_result][i]
    #         graph_list.append(g_data)
    #     return graph_list
    #
    # def write_to_file(self, path):
    #     if not (os.path.exists(path) and os.path.isfile(path)):
    #             os.makedirs(path)
    #     with open(path, "w") as outfile:
    #         assert len(train) == len(train_output) and len(test) == len(test_output)
    #         data = {}
    #         data["train"] = get_data_pairs(True)
    #         data["test"] = get_data_pairs(False)
    #         outfile.write(json.dumps(data, outfile))
