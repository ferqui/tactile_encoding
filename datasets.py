import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# helper functions
from scipy import signal, ndimage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

def load_data(file_name="./data/data_braille_letters_0.0.pkl", upsample_fac=1.0):
    '''
    Load the tactile Braille data.
    '''
    # TODO inlcude a dataloader (to open loading other data)
    data_dict = pd.read_pickle(file_name)  # 1kHz data (new dataset)

    # Extract data
    data = data_dict['taxel_data']
    labels = data_dict['letter']
    # TODO find a way to use letters as labels
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)  # labels as int numbers
    data_steps = len(data[0])

    # filter and resample
    resample_fac = 1  # value < 1 downsample, otherwise upsample
    data_resampled = []
    data_resampled_split = []
    filter_size = [int((data_steps*resample_fac)/75), 0]  # found manually
    max_val = 111.0/10  # found by iterating over data beforehand
    for _, trial in enumerate(data):
        if resample_fac == 1.0:
            # no resampling, just filtering (smoothing)
            data_dummy = ndimage.uniform_filter(
                trial, size=filter_size, mode='nearest')  # smooth

        elif resample_fac < 1.0:
            # downsampling and filtering
            data_dummy = signal.decimate(trial, int(
                1/resample_fac), axis=0)  # downsample
            data_dummy = ndimage.uniform_filter(
                data_dummy, size=filter_size, mode='nearest')  # smooth
        else:
            # upsampling and filtering
            data_dummy = signal.resample(
                trial, data_steps*resample_fac)  # upsample
            data_dummy = ndimage.uniform_filter(
                data_dummy, size=filter_size, mode='nearest')  # smooth

        # start at zero per sensor and normalize
        first_val = np.tile(
            data_dummy[int(data_steps*0.1)], (np.array(data_dummy).shape[0], 1))
        data_dummy = np.array((data_dummy - first_val)/max_val)

        # split each channel in two (positive, abs(negative))
        # positive on first 12 negative on last 12
        data_split = np.zeros(
            (np.array(data_dummy).shape[0], np.array(data_dummy).shape[1]*2))
        data_split[:, ::2] = np.where(data_dummy > 0, data_dummy, 0)
        data_split[:, 1::2] = abs(np.where(data_dummy < 0, data_dummy, 0))

        # discard the first 10% (not needed in the future when done in dataset creation)
        data_resampled.append(data_dummy[int(data_steps*0.1):])
        data_resampled_split.append(data_split[int(data_steps*0.1):])

    data_steps = len(data_resampled[0])  # update data steps

    data = torch.as_tensor(np.array(data_resampled), dtype=torch.float) 
    data_split = torch.as_tensor(np.array(data_resampled_split), dtype=torch.float)  # data to feed to encoding
    labels = torch.as_tensor(labels, dtype=torch.long)

    selected_chans = 2*len(data[0][0])
    return data_split, labels, selected_chans, data_steps, le, data

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
