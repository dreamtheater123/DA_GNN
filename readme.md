# Overview
This is an pytorch implementation of several data augmentation techniques for graph neural networks, including DropEdge, GAugM and an original data augmentation method called GAboC.</br>
This project aims to perform performance comparison of several graph data augmentation methods.
## About GAboC
Inspired by GAugM, which shows that the performance of graph neural networks can be further improved by adding "important" edges or deleting "unimportant" edges, we explore the possibility of using centrality to measure the importance of the edges. We found that using edge degree centrality to perform data augmentation can slightly improve the performance of graph neural networks.
## Graph neural networks
We choose to use GCN and GAT as the backbone GNN models.
## Datasets
We support Cora and Citeseer dataset in this project.
## Run the project
"main.py" is the entry file for this project. You can use the command
`python main.py`
to run the project.

You can also use the command
`python param_optim.py`
to do the parameter optimization.
## Project Structure
Directory "data" contains the downloaded datasets.<br>
Directory "data_augmentation" contains 4 graph data augmentation implementation code.<br>
Directory "model" contains the implementation code for GCN&GAT.<br>
**Config.py** contains all the hyperparameters of the model.<br>
**utils.py** contains some useful functions.<br>
**main.py** contains the pipeline of running the whole model.<br>
**param_optim.py** contains the code for parameter optimization. It optimizes the learning rate, hidden layer size, dropout rate, weight decay rate.
