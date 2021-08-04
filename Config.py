import torch
import os


class Config:
    def __init__(self):
        self.run_times = 1  # the number of repeated times of experiments
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU to run if it is available
        self.dataset = 'Cora'  # the name of the dataset
        self.GNN = 'GAT'  # GCN or GAT
        self.Data_Aug = 'original'  # the data augmentation method name: original or GAboC or GAboL or Dropedge or GAugM
        self.input_layer = 1433  # the input layer of the model (normally this is the number of features in each node)
        self.hidden_layer = 16  # the hidden layer of the model
        self.hidden_layer_GAT = 8  # size of the hidden layer of GAT
        self.output_layer = 7  # the output layer of the model
        self.epoch = 10000  # the number of epoch in the experiment
        self.lr_GCN = 0.001  # learning rate for GCN
        self.lr_GAT = 0.005  # learning rate for GAT
        self.percent = 0.2  # the percent of the added/removed edges
        self.hop = 2  # the number of hops for the edges to add
        self.early_stopping_round = 200  # the number of epochs for the model to stop training if no accuracy improvement
        self.enable_logging = True  # whether to produce log
        self.log_path = os.path.join('.', 'log', 'param_optim', self.GNN)  # path for producing the log
        self.num_heads = 8  # number of heads for multi-head attention
        self.weight_decay = 5e-4  # L2 regularization parameter
        self.dropout = 0.5  # dropout rate when performing forward propagation
        self.dropout_conv = 0.6  # dropout rate in GAT convolutional layer


config = Config()


if __name__ == '__main__':
    config = Config()
    print(config.log_path)
