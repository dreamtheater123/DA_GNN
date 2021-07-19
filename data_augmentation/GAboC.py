from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import torch_geometric.transforms as T
import networkx as nx
from networkx import edge_betweenness_centrality
import torch
import numpy as np

import matplotlib.pyplot as plt
from tqdm import *

def remove_edge(ori_graph_list, edge):
    # print()
    # print(edge)
    # print(ori_graph_list[0])
    # print(ori_graph_list[1])

    remove_edge_num = 0
    for index, first_node in enumerate(ori_graph_list[0]):
        if first_node == edge[0]:
            if ori_graph_list[1][index] == edge[1]:
                # print(index, ori_graph_list[0][index], ori_graph_list[1][index])
                ori_graph_list[0].pop(index)
                ori_graph_list[1].pop(index)
                # print(index, ori_graph_list[0][index], ori_graph_list[1][index])
                remove_edge_num = remove_edge_num + 1
    for index, first_node in enumerate(ori_graph_list[0]):
        if first_node == edge[1]:
            if ori_graph_list[1][index] == edge[0]:
                # print(index, ori_graph_list[0][index], ori_graph_list[1][index])
                ori_graph_list[0].pop(index)
                ori_graph_list[1].pop(index)
                # print(index, ori_graph_list[0][index], ori_graph_list[1][index])
                remove_edge_num = remove_edge_num + 1

    if remove_edge_num == 2:
        return ori_graph_list
    else:
        print('edge is not removed properly!')
        return False


def data_aug(data, dataG, percent):
    """
    this function performs the actual data augmentation operations.
    :param data: dataset in pyg form
    :param dataG: dataset in networkx form
    :param percent: the percent of modified edges
    :return: the pyg dataset after augmentation
    """
    dataG_EBC = edge_betweenness_centrality(dataG)
    sorted_EBC = sorted(dataG_EBC.items(), key=lambda d: d[1], reverse=False)
    num_removed_edges = int(len(sorted_EBC) * percent)
    print('number of removed edges by GAboC:', num_removed_edges)

    edge_np = data.edge_index.numpy()
    edge_list = edge_np.tolist()
    for i in tqdm(range(num_removed_edges)):
        edge_list = remove_edge(edge_list, sorted_EBC[i][0])
    print(len(edge_list[0]), len(edge_list[1]))
    edge_np = np.array(edge_list)

    return edge_np
