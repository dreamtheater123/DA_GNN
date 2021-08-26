# Overview
This is the code for project "Data Augmentation for Graph Neural Networks"
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

## Experiments to Run
The only experiment left to run is the GAboL on GAT using the split of 1208/500/1000.
Before running, you need to make sure the following parameters that are correctly set in **Config.py**.
```python
self.Cora_split = '1208'
self.GNN = 'GAT'
self.Data_Aug = 'GAboL'
```
Then, just run the **experiment.py**.
The key code in it is the copied in the following lines.
```python
for i in range(7, 16):
    # print(i)
    config.hop = i
    main()
```
It runs the experiment with the number of hops from 7 to 15.