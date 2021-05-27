import math
import torch_geometric
from typing import List
from torch_geometric.data import InMemoryDataset, download_url


def chunkify(_list, num_buckets):
    bucket_size = math.ceil(len(_list) / num_buckets)
    return [_list[bucket_size * i : bucket_size * (i + 1)] for i in range(num_buckets)]


def nx_to_pytorch_data(G, label=None):
    """Converts a networkx object into pytorch geometric data."""
    _G = torch_geometric.utils.from_networkx(G)
    if label is not None:
        print(label)
        _G.y = label
    return _G


def nx_to_pytorch_dataset(Gs: List, labels: List = None):
    """ Convert a list of networkx objects into pytorch geometric dataset """
    if not isinstance(Gs, list):
        Gs = [
            Gs,
        ]
    _dataset = [nx_to_pytorch_data(x, y) for x, y in zip(Gs, labels)]

    _data = [torch_geometric.utils.from_networkx(G) for G in Gs]
    if labels is not None:
        for G, label in zip(_data, labels):
            G.y = label
    return _data