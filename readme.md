# Overview
This is the code for project "Data Augmentation for Graph Neural Networks"
## Run the project
"main.py" is the entry file for this project. You can use the command
```
python main.py
```
to run the project.

You can also use the command
```
python param_optim.py
```
to do the parameter optimization.
## Project Structure
Directory "data" contains the downloaded datasets.<br>
Directory "data_augmentation" contains 4 graph data augmentation implementation code.<br>
Directory "model" contains the implementation code for GCN&GAT.<br>
Config.py contains all the hyperparameters of the model.
utils.py contains some useful functions.
main.py contains the pipeline of running the whole model.
param_optim.py contains the code for parameter optimization. It optimizes the learning rate, hidden layer size, dropout rate, weight decay rate.
