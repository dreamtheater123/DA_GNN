from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import torch_geometric.transforms as T
import networkx as nx
import torch
import numpy as np
import pickle
import os

from tqdm import *


def add_edge(ori_graph_list, edge):
    ori_graph_list[0].append(edge[0])
    ori_graph_list[1].append(edge[1])

    ori_graph_list[0].append(edge[1])
    ori_graph_list[1].append(edge[0])

    return ori_graph_list


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


def in_original_graph(ori_edge_list, edge):
    judge = False
    for index, first_node in enumerate(ori_edge_list[0]):
        if first_node == edge[0]:
            if ori_edge_list[1][index] == edge[1]:
                judge = True
                break

    return judge


def data_aug(data, dataG, percent):
    """
    this function performs the actual data augmentation operations.
    :param data: dataset in pyg form
    :param dataG: dataset in networkx form
    :param percent: the percent of modified edges
    :return: the pyg dataset after augmentation
    """
    dataE_numpy = data.edge_index.numpy()
    num_of_edges = dataE_numpy.shape[1]

    edge_prob_matrix = pickle.load(open('.//data//Cora_GAugM//cora_graph_2_logits.pkl', 'rb'))
    print('cora_graph_2_logits.pkl')
    print(edge_prob_matrix)

    existing_edge_prob_dict = {}
    missing_edge_prob_dict = {}
    del_edge_list = []
    for i in range(num_of_edges):  # create edge list
        if dataE_numpy[0][i] < dataE_numpy[1][i]:  # remove duplicate edges (as undirected edges are stored twice in pyg)
            existing_edge_prob_dict[(dataE_numpy[0][i], dataE_numpy[1][i])] = edge_prob_matrix[dataE_numpy[0][i]][
                dataE_numpy[1][i]]
            del_edge_list.append((dataE_numpy[0][i], dataE_numpy[1][i]))  # add all edges into this list
    # print(existing_edge_prob_dict)
    print('existing edges length in dataset:', len(existing_edge_prob_dict))

    for row in tqdm(range(edge_prob_matrix.shape[0])):
        for col in range(row + 1, edge_prob_matrix.shape[1]):
            missing_edge_prob_dict[(row, col)] = edge_prob_matrix[row][col]
    print('all possible edges length:', len(missing_edge_prob_dict))
    for del_edge in del_edge_list:  # if the edge is in the original graph, then removes it. That's how I derive the missing edges
        del missing_edge_prob_dict[del_edge]
    print('missing edges length in dataset:', len(missing_edge_prob_dict))

    # existing edge prob is from small to large, whereas missing edge prob is from large to small.
    sorted_existing_edge_prob_dict = sorted(existing_edge_prob_dict.items(), key=lambda d: d[1], reverse=False)
    sorted_missing_edge_prob_dict = sorted(missing_edge_prob_dict.items(), key=lambda d: d[1], reverse=True)

    num_processed_edges = int(len(sorted_existing_edge_prob_dict) * percent)
    edge_np = data.edge_index.numpy()
    edge_list = edge_np.tolist()
    # remove edges
    for i in tqdm(range(num_processed_edges)):
        edge_list = remove_edge(edge_list, sorted_existing_edge_prob_dict[i][0])
    print('edge number after removal:', len(edge_list[0]), len(edge_list[1]))
    # add edges
    for i in tqdm(range(num_processed_edges)):
        edge_list = add_edge(edge_list, sorted_missing_edge_prob_dict[i][0])
    print('edge number after addition:', len(edge_list[0]), len(edge_list[1]))
    edge_np = np.array(edge_list)

    return edge_np


if __name__ == '__main__':
    # data_aug(1, 1, 1)
    pass