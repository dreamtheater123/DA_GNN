from Config import config
import os
import matplotlib.pyplot as plt

config.Cora_split = '1208'
config.GNN = 'GAT'
config.Data_Aug = 'GAugM'
log_path = ''

if config.Data_Aug == 'original':
    log_path = os.path.join('.', 'log', 'Cora_' + config.Cora_split + '_500_1000', config.GNN, config.Data_Aug)
elif config.Data_Aug == 'GAboC' or config.Data_Aug == 'dropedge' or config.Data_Aug == 'GAugM':
    log_path = os.path.join('.', 'log', 'Cora_' + config.Cora_split + '_500_1000', config.GNN, config.Data_Aug)
elif config.Data_Aug == 'GAboL':
    log_path = os.path.join('.', 'log', 'Cora_' + config.Cora_split + '_500_1000', config.GNN, config.Data_Aug)
else:
    raise RuntimeError('Data augmentation method setting error!')
print(log_path)

dir_list = []
for root, dirs, files in os.walk(log_path, topdown=False):
    if root == log_path:
        dir_list = dirs
if dir_list == []:
    raise RuntimeError('dir_list setting error!')

print(dir_list)
dir_dict = {}
for index in range(len(dir_list)):
    dir_dict[float(dir_list[index])] = os.path.join(log_path, str(dir_list[index]), 'val_mean.txt')
print(dir_dict)

print()
x = []
y = []
for i in sorted(dir_dict):
    print(i, dir_dict[i])
    with open(dir_dict[i], 'r', encoding='utf-8') as f:
        lines = f.readlines()
        target = lines[-1].strip()
    print(target[26:])
    x.append(i)
    y.append(float(target[26:]))

print(x)
print(y)
title = config.Cora_split + ' + ' + config.GNN + ' + ' + config.Data_Aug
plt.plot(x, y)
plt.title(title)
if config.Data_Aug == 'GAboC' or config.Data_Aug == 'dropedge' or config.Data_Aug == 'GAugM':
    plt.xlabel('Percent of modified edges')
    plt.ylabel('Accuracy')
elif config.Data_Aug == 'GAboL':
    plt.xlabel('Number of hops')
    plt.ylabel('Accuracy')
# plt.savefig(os.path.join(log_path, title + '.jpg'))
plt.savefig(os.path.join('./log/visualization', title + '.jpg'))
plt.show()
