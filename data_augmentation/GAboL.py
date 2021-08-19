from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import torch_geometric.transforms as T
import networkx as nx
from networkx import edge_betweenness_centrality
import torch
import numpy as np

import matplotlib.pyplot as plt
from tqdm import *
import time


def find_trainmask_index(train_mask):
    """
    this function finds the split index for train_mask
    :param train_mask: the train_mask in a graph object
    :return: the split index for train_mask
    """
    split_indexes = []
    for index, item in enumerate(train_mask):
        if index == 0:  # train_mask[0] is True
            split_indexes.append(index)
        else:
            if train_mask[index] != train_mask[index-1]:
                split_indexes.append(index)

    return split_indexes

    # largest_index = 0
    # for index, item in enumerate(train_mask):
    #     if item == True:
    #         if index > largest_index:
    #             largest_index = index
    # return largest_index


def add_edge(ori_graph_list, edge):
    ori_graph_list[0].append(edge[0])
    ori_graph_list[1].append(edge[1])

    return ori_graph_list


def data_aug(data, dataG, hop):
    """
    this function performs the actual data augmentation operations.
    :param data: dataset in pyg form
    :param dataG: dataset in networkx form
    :param hop: the number of hops of the added edges
    :return: the pyg dataset after augmentation
    """
    tik = time.time()
    all_path = dict(nx.all_pairs_shortest_path(dataG, hop))
    tok = time.time()
    count = 0
    for item in all_path.values():
        count += len(item)
        # print(item)
    count /= 2
    print('number of qualified edges:', count)
    print('compute all shortest path cost time:', tok - tik)

    add_edge_list = []
    count = 0
    split_indexs = find_trainmask_index(data.train_mask)
    print('split indexs:', split_indexs)
    for i in range(2708):
        for key, item in all_path[i].items():  # edges like (1, 3) and (3, 1) are all added in this list
            if len(item) > 2:
                if dataG.has_edge(item[0], item[-1]) == False:  # 这里是找的最短路径为2的，所以肯定没有起点终点之间肯定没有edge
                    if data.y[item[0]].item() == data.y[item[-1]].item():
                        if len(split_indexs) == 2:
                            if split_indexs[0] <= item[0] < split_indexs[1] or split_indexs[0] <= item[-1] < split_indexs[1]:
                                add_edge_list.append([item[0], item[-1]])
                                count += 1
                        elif len(split_indexs) == 4:
                            if split_indexs[0] <= item[0] < split_indexs[1] or split_indexs[0] <= item[-1] < split_indexs[1] or split_indexs[2] <= item[0] < split_indexs[3] or split_indexs[2] <= item[-1] < split_indexs[3]:
                                add_edge_list.append([item[0], item[-1]])
                                count += 1
                        else:
                            raise RuntimeError('GAboL training set split index error!')
    # print(add_edge_list)
    print('number of added edges (double check):', count, len(add_edge_list))
    print('number of hops:', hop)

    edge_np = data.edge_index.numpy()
    edge_list = edge_np.tolist()
    for edge in tqdm(add_edge_list):
        edge_list = add_edge(edge_list, edge)
    print('double check the shape of the edge matrix:', len(edge_list[0]), len(edge_list[1]))
    edge_np = np.array(edge_list)

    return edge_np


if __name__ == '__main__':
    dataset = Planetoid(root='./../data/Cora', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    find_trainmask_index(data.train_mask)