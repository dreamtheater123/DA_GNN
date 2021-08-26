from main import main
from Config import config

for i in range(7, 16):
    # print(i)
    config.hop = i
    main()
