from main import main
import numpy as np
import random
from Config import config
import os

if __name__ == '__main__':
    print('parameter optimization will perform repeated experiments for', config.run_times_PO, 'times.')
    for repeat in range(config.run_times_PO):
        # pre-configure
        lr = np.random.rand()
        lr = -2 * lr - 2  # range: [-4, -2]
        lr = 10 ** lr

        hidden_layer_list = []
        for i in range(10, 501):
            hidden_layer_list.append(i)
        hidden_layer = random.choice(hidden_layer_list)

        dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        dropout = random.choice(dropout_list)

        weight_decay = np.random.rand()
        weight_decay = -2 * weight_decay - 2  # range: [-4, -2]
        weight_decay = 10 ** weight_decay

        # settings
        config.GNN = 'GAT'
        config.log_path = os.path.join('.', 'log', 'param_optim', config.GNN)  # important!!!
        if config.GNN == 'GCN':
            config.lr_GCN = lr
            config.hidden_layer_GCN = hidden_layer
            config.dropout = dropout
            config.weight_decay = weight_decay

            print('GNN:', config.GNN)
            print('learning rate:', config.lr_GCN)
            print('hidden layer size:', config.hidden_layer_GCN)
            print('dropout rate:', config.dropout)
            print('weight decay rate:', config.weight_decay)
        elif config.GNN == 'GAT':
            config.lr_GAT = lr
            config.hidden_layer_GAT = hidden_layer
            config.dropout = dropout
            config.weight_decay = weight_decay

            print('GNN:', config.GNN)
            print('learning rate:', config.lr_GAT)
            print('hidden layer size:', config.hidden_layer_GAT)
            print('dropout rate:', config.dropout)
            print('weight decay rate:', config.weight_decay)
        else:
            raise RuntimeError('config.GNN setting error!')
        main()
