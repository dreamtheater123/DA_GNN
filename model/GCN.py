import os.path as osp
from tqdm import *

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from Config import Config


class GCN(torch.nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(config.input_layer, 16, cached=True,
                             normalize=True)
        self.conv2 = GCNConv(16, config.output_layer, cached=True,
                             normalize=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, this_data):
        x, edge_index, edge_weight = this_data.x, this_data.edge_index, this_data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    config = Config()
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', config.dataset)
    print(path)
    dataset = Planetoid(path, config.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    config.input_layer = dataset.num_features
    config.output_layer = dataset.num_classes
    print(dataset.num_features, dataset.num_classes)
    print(config.input_layer, config.output_layer)
    model = GCN(config).to(config.device)
