import torch
import os

train_dir = os.listdir('train/')
filename = '../../Imagenet32_Scripts/map_clsloc.txt'

summand = 0
for line in open(filename):
    wn_id = line.split()[0]
    print('ID: ', wn_id)

    if wn_id in train_dir:
        summand += 1

print('true' if summand == 1000 else str(1000 - summand))
