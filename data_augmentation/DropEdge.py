from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import torch_geometric.transforms as T
import networkx as nx
import torch
import numpy as np
from tqdm import *
import random
from Config import Config


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
    config = Config()
    num_removed_edges = int(dataG.number_of_edges() * percent)
    print('number of removed edges:', num_removed_edges)

    edge_list_epochs = []
    edge_np = data.edge_index.numpy()
    for i in tqdm(range(config.epoch)):
        edge_list = edge_np.tolist()
        # print(len(edge_list[0]), len(edge_list[1]))
        for i in range(num_removed_edges):  # tqdm has disabled
            index = random.randint(0, len(edge_list[0]) - 1)
            edge_list = remove_edge(edge_list, [edge_list[0][index], edge_list[1][index]])
        edge_list_epochs.append(edge_list)
        print(len(edge_list[0]), len(edge_list[1]))
    print('\n', np.array(edge_list_epochs).shape)

    return edge_list_epochs


def graph_augmentation(percent):
    dataset = Planetoid(root='./../data/Cora', name='Cora', transform=T.NormalizeFeatures())
    # print(len(dataset))
    # print(type(dataset))
    data = dataset[0]
    # print(data.edge_index)
    # print(type(data.train_mask))
    # print(data.train_mask.sum())
    # print(data)
    # print("now let's print the train mask of cora dataset: ")
    # print(data.train_mask)
    # print(len(data.train_mask))  # train mask是一个列表，所有train的node都标记为true。反之为false
    # print('cora is directed: ' + str(data.is_directed()))
    # print(data.x.requires_grad, data.edge_index.requires_grad)

    dataX_numpy = data.x.numpy()
    dataE_numpy = data.edge_index.numpy()
    # print(dataX_numpy.shape[0])
    # print(dataE_numpy)

    coraG_nodes = []
    coraG_edges = []
    num_of_nodes = dataX_numpy.shape[0]
    num_of_edges = dataE_numpy.shape[1]

    for i in range(num_of_nodes):  # create node list
        coraG_nodes.append(i)
    for i in range(num_of_edges):  # create edge list
        coraG_edges.append((dataE_numpy[0][i], dataE_numpy[1][i]))

    coraG = nx.Graph()
    coraG.add_nodes_from(coraG_nodes)
    coraG.add_edges_from(coraG_edges)
    # print('number of nodes of coraG: ' + str(coraG.number_of_nodes()))
    # print('number of edges of coraG: ' + str(coraG.number_of_edges()))

    num_removed_edges = int(coraG.number_of_edges() * percent)
    # print(num_removed_edges)

    edge_list_200 = []
    edge_np = data.edge_index.numpy()
    for i in tqdm(range(200)):
        edge_list = edge_np.tolist()
        # print(len(edge_list[0]), len(edge_list[1]))
        for i in range(num_removed_edges):  # tqdm has disabled
            index = random.randint(0, len(edge_list[0]) - 1)
            edge_list = remove_edge(edge_list, [edge_list[0][index], edge_list[1][index]])
        edge_list_200.append(edge_list)
        print(len(edge_list[0]), len(edge_list[1]))
    print('\n', np.array(edge_list_200).shape)

    data_new_list = []
    for el in edge_list_200:
        x_new = data.x.clone()
        edge_index_new = torch.tensor(el, dtype=torch.long)
        test_mask_new = data.test_mask.clone()
        train_mask_new = data.train_mask.clone()
        val_mask_new = data.val_mask.clone()
        y_new = data.y.clone()
        data_new = Data(x=x_new, edge_index=edge_index_new, test_mask=test_mask_new, train_mask=train_mask_new,
                        val_mask=val_mask_new, y=y_new)
        data_new_list.append(data_new)


    return data_new_list

    # nx.draw(coraG, with_labels = True)
    # plt.show()

if __name__ == '__main__':
    data_new_list = graph_augmentation(0.5)