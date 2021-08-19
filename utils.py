import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from Config import config
import networkx as nx
import torch
from torch_geometric.data import Data
import os
import datetime
from numpy import mean


def load_dataset(dataset_name):
    print('loading dataset...')
    if dataset_name == 'Cora':
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset_name)
        print('from path:', path)
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
    else:
        raise RuntimeError('Cannot found dataset "' + str(dataset_name) + '"')

    return dataset


def from_pyg_to_networkx(data):
    """
    this function converts the dataset of pyg form into networkx form
    while pyg stands for 'pytorch geometric'
    :return:
    """
    dataX_numpy = data.x.numpy()
    dataE_numpy = data.edge_index.numpy()
    dataG_nodes, dataG_edges = [], []
    num_of_nodes = dataX_numpy.shape[0]
    num_of_edges = dataE_numpy.shape[1]
    for i in range(num_of_nodes):  # create node list
        dataG_nodes.append(i)
    for i in range(num_of_edges):  # create edge list
        dataG_edges.append((dataE_numpy[0][i], dataE_numpy[1][i]))
    dataG = nx.Graph()
    dataG.add_nodes_from(dataG_nodes)
    dataG.add_edges_from(dataG_edges)
    print('number of nodes of networkx: ' + str(dataG.number_of_nodes()))
    print('number of edges of networkx: ' + str(dataG.number_of_edges()))

    return dataG


def from_numpy_edge_to_pyg(data, edge_np):
    """
    this function make the augmented pyg dataset from the modified edges, which is in numpy form
    :return:
    """
    x_new = data.x.clone()
    edge_index_new = torch.tensor(edge_np, dtype=torch.long)
    test_mask_new = data.test_mask.clone()
    train_mask_new = data.train_mask.clone()
    val_mask_new = data.val_mask.clone()
    y_new = data.y.clone()
    data_new = Data(x=x_new, edge_index=edge_index_new, test_mask=test_mask_new, train_mask=train_mask_new,
                    val_mask=val_mask_new, y=y_new)  # x = data.x, the previous version is x = x_new
    print('comparison: ')
    print(data)
    print(data_new)

    return data_new


def saving_log_PO(log_list, val_acc, test_acc, lr, hidden_layer, dropout, weight_decay, GNN):
    with open(os.path.join(config.log_path, 'acc_collection.csv'), 'a+', encoding='utf-8') as f:
        f.write(','.join([str(val_acc), str(test_acc), str(lr), str(hidden_layer), str(dropout), str(weight_decay), str(GNN)]) + '\n')
    # find the total length in acc_collection file
    with open(os.path.join(config.log_path, 'acc_collection.csv'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    file_len = len(lines)
    with open(os.path.join(config.log_path, 'line' + str(file_len) + '.txt'), 'w', encoding='utf-8') as f:
        f.writelines(log_list)


def saving_log_normal(log_list, val_acc, test_acc):
    # print(os.path.join(config.log_path, datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S') + '.txt'))
    with open(os.path.join(config.log_path, datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S').replace(':', '_') + '.txt'), 'w', encoding='utf-8') as f:
        f.writelines(log_list)
        f.writelines(['Best validation accuracy: ' + str(val_acc) + '\n', 'Best test accuracy: ' + str(test_acc) + '\n'])


def saving_log_mean(val_list):
    with open(os.path.join(config.log_path, 'val_mean.txt'), 'w', encoding='utf-8') as f:
        for val in val_list:
            f.write(str(val) + '\n')
        f.write('mean validation accuracy: ' + str(mean(val_list)) + '\n')


def change_split(data):
    index_list = list(range(data.train_mask.shape[0]))
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        for index, item in enumerate(mask):
            if item == True:
                index_list.remove(index)
    for index in index_list:
        data.train_mask[index] = True

    return data


if __name__ == '__main__':
    # cora = load_dataset(config.dataset)
    # dataG = from_pyg_to_networkx(cora[0])
    # print(dataG)
    # # print('success!')
    saving_log_normal(['1\n', '2\n'], 11, 111)
