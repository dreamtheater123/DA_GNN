import os.path as osp
from tqdm import *

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
# from Config import Config


class GCN(torch.nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(config.input_layer, config.hidden_layer_GCN, cached=False,
                             normalize=True)
        self.conv2 = GCNConv(config.hidden_layer_GCN, config.output_layer, cached=False,
                             normalize=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, this_data, config):
        x, edge_index, edge_weight = this_data.x, this_data.edge_index, this_data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=config.dropout_GCN, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    pass
