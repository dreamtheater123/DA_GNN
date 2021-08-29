from main import main
from utils import load_dataset, change_split
from Config import config
import torch
import os
import time
import datetime

# for i in [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     config.percent = i
#     main()
#     # , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

for i in range(7, 16):
    # print(i)
    config.hop = i
    main()

# for i in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     config.percent = i
#     main()

# config.percent = 1.0
# main()