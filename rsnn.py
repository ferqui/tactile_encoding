"""
Here all for the RSNN from tactile Braille reading is set up.
"""

import warnings
import os
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sn

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

def run_snn():
    global use_trainable_out
    use_trainable_out = False
    global use_trainable_tc
    use_trainable_tc = False
    global use_dropout
    use_dropout = False
    global batch_size
    batch_size = 128
    global lr
    lr = 0.0015

    # set the number of epochs you want to train the network (default = 300)
    epochs = 50
    save_fig = True  # set True to save the plots

