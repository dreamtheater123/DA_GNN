import os.path as osp
from tqdm import *

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
# from Config import Config


class GAT(torch.nn.Module):
    def __init__(self, config):
        super(GAT, self).__init__()
        self.conv1 = GATConv(config.input_layer, config.hidden_layer_GAT, heads=config.num_heads, dropout=config.dropout_GAT)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(config.hidden_layer_GAT * config.num_heads, config.output_layer, heads=1, concat=False,
                             dropout=config.dropout_GAT)

    def forward(self, this_data, config):
        x = F.dropout(this_data.x, p=config.dropout_GAT, training=self.training)
        x = F.elu(self.conv1(x, this_data.edge_index))
        x = F.dropout(x, p=config.dropout_GAT, training=self.training)
        x = self.conv2(x, this_data.edge_index)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    pass
