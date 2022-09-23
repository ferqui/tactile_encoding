import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# helper functions
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader


def load_analog_data(file_name):

    data_dict = pd.read_pickle(file_name)

    letter_written = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                      'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # Extract data
    nb_repetitions = 200
    data = data_dict['taxel_data']
    labels = data_dict['letter']
    # find a way to use letters as labels
    le = LabelEncoder()
    le.fit(labels)
    # labels['categorical_label'] = le.transform(labels)
    labels = le.transform(labels) # labels as int numbers

    data_steps = len(data[0])

    # Upsample
    upsample_fac = 10
    data_upsampled = []
    for _, trial in enumerate(data):
        data_upsampled.append(signal.resample(trial, data_steps*upsample_fac))
    data_steps = len(data_upsampled[0]) # update data steps

    # convert into tensors
    data = torch.as_tensor(data_upsampled, dtype=torch.float)
    labels = torch.as_tensor(labels, dtype=torch.long)

    # # Select nonzero inputs
    selected_chans = len(data[0][0])  # read out from data

    # Standardize data
    rshp = data.reshape((-1, data.shape[2]))
    data = (data-rshp.mean(0))/(rshp.std(0)+1e-3)

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels)

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)

    return ds_train, ds_test, labels, selected_chans, data_steps


def load_event_data(params, file_name):

    letter_written = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                      'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    max_time = int(51*25)  # ms
    time_bin_size = int(params['time_bin_size'])  # ms
    global time
    time = range(0, max_time, time_bin_size)

    global time_step
    time_step = time_bin_size*0.001
    data_steps = len(time)

    data_dict = pd.read_pickle(file_name)

    # Extract data
    data = []
    labels = []
    bins = 1000  # ms conversion
    nchan = len(data_dict['events'][1])  # number of channels per sensor
    # loop over all trials
    for i, sample in enumerate(data_dict['events']):
        events_array = np.zeros(
            [nchan, round((max_time/time_bin_size)+0.5), 2])
        # loop over sensors (taxel)
        for taxel in range(len(sample)):
            # loop over On and Off channels
            for event_type in range(len(sample[taxel])):
                if sample[taxel][event_type]:
                    indx = bins*(np.array(sample[taxel][event_type]))
                    indx = np.array((indx/time_bin_size).round(), dtype=int)
                    events_array[taxel, indx, event_type] = 1
        events_array = np.reshape(np.transpose(
            events_array, (1, 0, 2)), (events_array.shape[1], -1))
        selected_chans = 2*nchan
        data.append(events_array)
        labels.append(letter_written.index(data_dict['letter'][i]))

    # return data,labels
    data = np.array(data)
    labels = np.array(labels)

    data = torch.tensor(data, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.20, shuffle=True, stratify=labels)

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)

    return ds_train, ds_test, labels, selected_chans, data_steps
