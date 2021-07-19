import torch


class Config:
    def __init__(self):
        self.run_times = 1  # the number of repeated times of experiments
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU to run if it is available
        self.dataset = 'Cora'  # the name of the dataset
        self.GNN = 'GCN'  # GCN or GAT
        self.Data_Aug = 'original'  # the data augmentation method name: original or GAboC or GAboL or Dropedge or GAugM
        self.input_layer = 1433  # the input layer of the model (normally this is the number of features in each node)
        self.hidden_layer = 16  # the hidden layer of the model
        self.output_layer = 7  # the output layer of the model
        self.epoch = 200  # the number of epoch in the experiment
        self.percent = 0.2  # the percent of the added/removed edges
        self.hop = 2  # the number of hops for the edges to add
