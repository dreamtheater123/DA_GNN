import random

import numpy as np
from tqdm import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from utils import load_dataset, from_pyg_to_networkx, from_numpy_edge_to_pyg, saving_log_PO, saving_log_normal, saving_log_mean, change_split
from Config import config
import os


def train(optimizer, model, data_loader):
    model.train()
    total_loss = 0
    total_examples = 0
    for graph in data_loader:
        optimizer.zero_grad()
        loss = F.nll_loss(model(graph, config)[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * graph.num_nodes
        total_examples += graph.num_nodes

    return total_loss / total_examples


def test(model, test_data):
    model.eval()
    logits, accs = model(test_data, config), []
    for _, mask in test_data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(test_data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def main():
    # determine 'config.log_path' folder and create the folder if not exists
    if config.enable_logging == 'PO':
        config.log_path = os.path.join('.', 'log', 'param_optim', config.GNN)  # path for producing the log
    elif config.enable_logging == 'normal':
        if config.Data_Aug == 'original':
            config.log_path = os.path.join('.', 'log', 'Cora_' + config.Cora_split + '_500_1000', config.GNN, config.Data_Aug)
        elif config.Data_Aug == 'GAboC' or config.Data_Aug == 'dropedge' or config.Data_Aug == 'GAugM':
            config.log_path = os.path.join('.', 'log', 'Cora_' + config.Cora_split + '_500_1000', config.GNN, config.Data_Aug,
                                         str(config.percent))
        elif config.Data_Aug == 'GAboL':
            config.log_path = os.path.join('.', 'log', 'Cora_' + config.Cora_split + '_500_1000', config.GNN, config.Data_Aug, str(config.hop))
        else:
            raise RuntimeError('Data augmentation method setting error!')
        if not os.path.exists(config.log_path):
            os.makedirs(config.log_path)

    # loading dataset
    if config.dataset == 'Cora':
        dataset = load_dataset(config.dataset)
        data = dataset[0]
        if config.Cora_split == '1208':
            data = change_split(data)
        elif config.Cora_split == '140':
            pass
        else:
            raise RuntimeError('Cora dataset does not support this split.')
        dataG = from_pyg_to_networkx(data)
    else:
        raise RuntimeError('Cannot found dataset "' + str(config.dataset) + '"')
    config.input_layer = dataset.num_features
    config.output_layer = dataset.num_classes

    test_data = None
    DA_flag = 0  # the flag will be 1 if the data augmentation method is dropedge
    if config.Data_Aug == 'dropedge':  # the dataset settings of dropedge is a bit different from other methods.
        DA_flag = 1
        from data_augmentation.DropEdge import data_aug
        edge_np = data_aug(data, dataG, config.percent)
        data_list = []
        for i in range(config.epoch):
            data_list.append(from_numpy_edge_to_pyg(data, edge_np[i]))  # .to(config.device)
        test_data = data.to(config.device)
    else:
        if config.Data_Aug == 'original':
            data_loader = [data]
            test_data = data.to(config.device)
        elif config.Data_Aug == 'GAboC':
            from data_augmentation.GAboC import data_aug
            edge_np = data_aug(data, dataG, config.percent)
            data_new = from_numpy_edge_to_pyg(data, edge_np)
            data_loader = [data, data_new]
            test_data = data.to(config.device)
        elif config.Data_Aug == 'GAboL':
            from data_augmentation.GAboL import data_aug
            edge_np = data_aug(data, dataG, config.hop)
            data_new = from_numpy_edge_to_pyg(data, edge_np)
            data_loader = [data_new]
            test_data = data_new.to(config.device)
        elif config.Data_Aug == 'GAugM':
            from data_augmentation.GAugM import data_aug
            edge_np = data_aug(data, dataG, config.percent)
            data_new = from_numpy_edge_to_pyg(data, edge_np)
            data_loader = [data_new]
            test_data = data_new.to(config.device)
        else:
            if DA_flag == 0:
                raise RuntimeError('Data_Aug parameter setting error.')

        for index in range(len(data_loader)):  # 修改：把训练的和测试的数据的cuda指针弄到一起
            data_loader[index] = data_loader[index].to(config.device)
        # for GAboL exclusively
        if config.Data_Aug == 'GAboL':
            test_data = data_loader[0]
        data_loader = DataLoader(data_loader, batch_size=1)

    best_val_list = []
    for times in range(config.run_times):  # the number of repeated experiments
        torch.cuda.empty_cache()
        # loading GNN model
        if config.GNN == 'GCN':
            from model.GCN import GCN
            model = GCN(config).to(config.device)
            optimizer = torch.optim.Adam([
                dict(params=model.conv1.parameters(), weight_decay=config.weight_decay_GCN),
                dict(params=model.conv2.parameters(), weight_decay=0)
            ], lr=config.lr_GCN)  # Only perform weight-decay on first convolution.
        elif config.GNN == 'GAT':
            from model.GAT import GAT
            model = GAT(config).to(config.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_GAT, weight_decay=config.weight_decay_GAT)
        else:
            raise RuntimeError('Cannot found GNN model "' + str(config.GNN) + '"')
        print(model.parameters)
        # # model parameters initialization
        # for name, w in model.named_parameters():
        #     print(name, w.shape)
        #     if 'bias' in name:
        #         nn.init.normal_(w)
        #     else:
        #         nn.init.xavier_normal_(w)
        for i in range(10):
            torch.cuda.empty_cache()

        best_val_acc = 0
        best_test_acc = 0
        best_epoch = 0
        last_epoch = None
        flag = 0
        log_list = []
        for epoch in range(1, config.epoch + 1):
            if config.Data_Aug == 'dropedge':
                data_loader = [data_list[epoch - 1].to(config.device)]
                data_loader = DataLoader(data_loader, batch_size=1)
            loss = train(optimizer, model, data_loader)
            train_acc, tmp_val_acc, tmp_test_acc = test(model, test_data)
            # free the cuda memory space
            if config.Data_Aug == 'dropedge':
                data_list[epoch - 1].to(torch.device('cpu'))
            if tmp_val_acc > best_val_acc:
                best_val_acc = tmp_val_acc
                best_epoch = epoch
            if tmp_test_acc > best_test_acc:
                best_test_acc = tmp_test_acc
            log = 'Epoch: {:03d}, Loss: {:.6f}, Train: {:.6f}, Val: {:.6f}, Test: {:.6f}, Val_acc_best: {:.6f}, Test_acc_best: {:.6f}'.format(
                epoch, loss, train_acc, tmp_val_acc, tmp_test_acc, best_val_acc, best_test_acc)
            log_list.append(log + '\n')
            print(log)
            if epoch - best_epoch >= config.early_stopping_round:  # early stopping
                flag = 1
                last_epoch = epoch
                break
        print()
        if flag == 1:
            if last_epoch is None:
                raise RuntimeError('Accounting error when setting the value of last_epoch.')
            print('early stopping at epoch', last_epoch)
        print('best validation accuracy:', best_val_acc)
        print('best testing accuracy:', best_test_acc)

        if config.enable_logging == 'PO':
            if config.GNN == 'GCN':
                saving_log_PO(log_list, best_val_acc, best_test_acc, config.lr_GCN, config.hidden_layer_GCN,
                           config.dropout_GCN,
                           config.weight_decay_GCN, config.GNN)
            elif config.GNN == 'GAT':
                saving_log_PO(log_list, best_val_acc, best_test_acc, config.lr_GAT, config.hidden_layer_GAT,
                           config.dropout_GAT,
                           config.weight_decay_GAT, config.GNN)
        elif config.enable_logging == 'normal':
            saving_log_normal(log_list, best_val_acc, best_test_acc)

        best_val_list.append(best_val_acc)

        # # map the old model to cpu memory
        # model.to(torch.device('cpu'))

    if config.enable_logging == 'normal':
        saving_log_mean(best_val_list)


if __name__ == '__main__':
    main()
