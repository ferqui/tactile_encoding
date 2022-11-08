import os
import pickle
import numpy as np
import pandas as pd

# init dataloading (load the last trial)
file_storage_found = False
idx_file_storage = 1
while not file_storage_found:
    file_storage_path = './results/record_' + str(idx_file_storage) + '.pkl'
    if os.path.isfile(file_storage_path):
        idx_file_storage += 1
    else:
        file_storage_found = True

# put id here if NOT last should be loaded
file_storage_path = './results/record_' + str(idx_file_storage-1) + '.pkl'

with open(file_storage_path, 'rb') as f:
    data = pickle.load(f)

print("BREAK")