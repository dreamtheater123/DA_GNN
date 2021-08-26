import torch
import os


class Config:
    def __init__(self):
        self.run_times = 10  # the number of repeated times of experiments
        self.run_times_PO = 500  # the number of repeated times in parameter optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU to run if it is available
        self.dataset = 'Cora'  # the name of the dataset
        self.Cora_split = '1208'  # the size of the Cora training set (140 or 1208)
        self.GNN = 'GAT'  # GCN or GAT
        self.Data_Aug = 'GAboL'  # the data augmentation method name: original or GAboC or GAboL or Dropedge or GAugM
        self.input_layer = 1433  # the input layer of the model (normally this is the number of features in each node)
        self.hidden_layer_GCN = 442  # the hidden layer of the model
        self.hidden_layer_GAT = 75  # size of the hidden layer of GAT
        self.output_layer = 7  # the output layer of the model
        self.epoch = 1000  # the number of epoch in the experiment
        self.lr_GCN = 0.00842  # learning rate for GCN
        self.lr_GAT = 0.002246  # learning rate for GAT
        self.percent = 0.02  # the percent of the added/removed edges
        self.hop = 2  # the number of hops for the edges to add
        self.early_stopping_round = 200  # the number of epochs for the model to stop training if no accuracy improvement
<<<<<<< HEAD
        self.enable_logging = 'normal'  # whether to produce log and to produce which kind of log (False, PO, normal)
=======
        self.enable_logging = 'no'  # whether to produce log and to produce which kind of log (False, PO, normal)
>>>>>>> 59b2b43afa4fe720940850043316a4a76a22c0ab
        self.num_heads = 8  # number of heads for multi-head attention
        self.weight_decay_GCN = 0.003276  # L2 regularization parameter for GCN
        self.weight_decay_GAT = 0.000107  # L2 regularization parameter for GAT
        self.dropout_GCN = 0.6  # dropout rate when performing forward propagation
        self.dropout_GAT = 0.8  # dropout rate for GAT

        if self.enable_logging == 'PO':
            self.log_path = os.path.join('.', 'log', 'param_optim', self.GNN)  # path for producing the log
        elif self.enable_logging == 'normal':
            if self.Data_Aug == 'original':
                self.log_path = os.path.join('.', 'log', 'Cora_' + self.Cora_split + '_500_1000', self.GNN, self.Data_Aug)
            elif self.Data_Aug == 'GAboC' or self.Data_Aug == 'dropedge' or self.Data_Aug == 'GAugM':
                self.log_path = os.path.join('.', 'log', 'Cora_' + self.Cora_split + '_500_1000', self.GNN, self.Data_Aug, str(self.percent))
            elif self.Data_Aug == 'GAboL':
                self.log_path = os.path.join('.', 'log', 'Cora_' + self.Cora_split + '_500_1000', self.GNN, self.Data_Aug, str(self.hop))
            else:
                raise RuntimeError('Data augmentation method setting error!')
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)


config = Config()


if __name__ == '__main__':
    print(config.log_path)
