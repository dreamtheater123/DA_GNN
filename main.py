from tqdm import *
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from utils import load_dataset, from_pyg_to_networkx, from_numpy_edge_to_pyg
from Config import Config

config = Config()


def train(optimizer, model, data_loader):
    model.train()
    total_loss = 0
    total_examples = 0
    for graph in data_loader:
        optimizer.zero_grad()
        loss = F.nll_loss(model(graph)[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * graph.num_nodes
        total_examples += graph.num_nodes

    return total_loss / total_examples


def test(model, test_data):
    model.eval()
    logits, accs = model(test_data), []
    for _, mask in test_data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(test_data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def main():
    # loading dataset
    if config.dataset == 'Cora':
        dataset = load_dataset(config.dataset)
        data = dataset[0]
        dataG = from_pyg_to_networkx(data)
    else:
        raise RuntimeError('Cannot found dataset "' + str(config.dataset) + '"')
    config.input_layer = dataset.num_features
    config.output_layer = dataset.num_classes

    if config.Data_Aug == 'DropEdge':  # the dataset settings of DropEdge is a bit different from other methods.
        from data_augmentation.DropEdge import data_aug
        edge_np = data_aug(data, dataG, config.percent)
        data_list = []
        for i in range(config.epoch):
            data_list.append(from_numpy_edge_to_pyg(data, edge_np[i]).to(config.device))
    else:
        if config.Data_Aug == 'original':
            data_loader = [data]
        elif config.Data_Aug == 'GAboC':
            from data_augmentation.GAboC import data_aug
            edge_np = data_aug(data, dataG, config.percent)
            data_new = from_numpy_edge_to_pyg(data, edge_np)
            data_loader = [data, data_new]
        elif config.Data_Aug == 'GAboL':
            from data_augmentation.GAboL import data_aug
            edge_np = data_aug(data, dataG, config.hop)
            data_new = from_numpy_edge_to_pyg(data, edge_np)
            data_loader = [data_new]
        elif config.Data_Aug == 'GAugM':
            from data_augmentation.GAugM import data_aug
            edge_np = data_aug(data, dataG, config.percent)
            data_new = from_numpy_edge_to_pyg(data, edge_np)
            data_loader = [data_new]
        else:
            raise RuntimeError('Data_Aug parameter setting error.')

        for index in range(len(data_loader)):
            data_loader[index] = data_loader[index].to(config.device)
        data_loader = DataLoader(data_loader, batch_size=1)

    for times in range(config.run_times):  # the number of repeated experiments
        # loading GNN model
        if config.GNN == 'GCN':
            from model.GCN import GCN
            model = GCN(config).to(config.device)
            optimizer = torch.optim.Adam([
                dict(params=model.conv1.parameters(), weight_decay=5e-4),
                dict(params=model.conv2.parameters(), weight_decay=0)
            ], lr=0.01)  # Only perform weight-decay on first convolution.
        elif config.GNN == 'GAT':
            from model.GAT import GAT
            model = GAT(config).to(config.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        else:
            raise RuntimeError('Cannot found GNN model "' + str(config.GNN) + '"')

        best_test_acc = 0
        for epoch in range(1, config.epoch + 1):
            if config.Data_Aug == 'DropEdge':
                data_loader = [data_list[epoch - 1]]
                data_loader = DataLoader(data_loader, batch_size=1)
            loss = train(optimizer, model, data_loader)
            train_acc, val_acc, tmp_test_acc = test(model, data)
            if tmp_test_acc > best_test_acc:
                best_test_acc = tmp_test_acc
            log = 'Epoch: {:03d}, Loss: {:.6f}, Train: {:.6f}, Val: {:.6f}, Test: {:.6f}, Test_acc_best: {:.6f}'
            print(log.format(epoch, loss, train_acc, val_acc, tmp_test_acc, best_test_acc))
        print('best testing accuracy:', best_test_acc)


if __name__ == '__main__':
    main()