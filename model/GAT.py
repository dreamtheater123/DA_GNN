import os.path as osp
from tqdm import *

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from Config import Config


class GAT(torch.nn.Module):
    def __init__(self, config):
        super(GAT, self).__init__()
        self.conv1 = GATConv(config.input_layer, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, config.output_layer, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, this_data):
        x = F.dropout(this_data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, this_data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, this_data.edge_index)
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
    model = GAT(config).to(config.device)
