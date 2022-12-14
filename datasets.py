import torch
import pandas as pd
import numpy as np
# import random
import matplotlib.pyplot as plt

# helper functions
from scipy import signal, ndimage
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset #, DataLoader

def load_analog_data(file_name, upsample_fac, specify_letters = []):

    data_dict = pd.read_pickle(file_name)
    # print(data_dict)
    # Extract data
    if len(specify_letters) != 0:
        letter_list = []
        for letter in specify_letters:
            letter_list.append(np.where(data_dict['letter'] == letter)[0])
            labels = data_dict['letter'][np.array(letter_list).flatten()]
            data = data_dict['taxel_data'][np.array(letter_list).flatten()]
    else:
        data = data_dict['taxel_data']
        labels = data_dict['letter']
    # find a way to use letters as labels
    le = LabelEncoder()
    le.fit(labels)
    # labels['categorical_label'] = le.transform(labels)
    labels = le.transform(labels) # labels as int numbers

    data_steps = len(data[0])

    # Upsample
    data_upsampled = []
    for _, trial in enumerate(data):
        data_upsampled.append(signal.resample(trial, data_steps*upsample_fac))
    data_steps = len(data_upsampled[0]) # update data steps

    # convert into tensors
    data = torch.as_tensor(np.array(data_upsampled), dtype=torch.float)
    labels = torch.as_tensor(labels, dtype=torch.long)

    # # Select nonzero inputs
    selected_chans = len(data[0][0])  # read out from data

    # Standardize data
    rshp = data.reshape((-1, data.shape[2]))
    data = (data-rshp.mean(0))/(rshp.std(0)+1e-3)

    x_train, x_test, y_train, y_test = train_test_split(
        data.cpu(), labels.cpu(), test_size=0.2, shuffle=True, stratify=labels.cpu())

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)

    return ds_train, ds_test, labels, selected_chans, data_steps

letters = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
dtype = torch.float

def load_and_extract_events(params, file_name, ratios=[0.8, 0, 0.2], taxels=None, letter_written=letters):
    if np.sum(ratios) != 1:
        raise ValueError('Check the correct ratios are used: got sum > 1')

    # max_time = int(54*25) #ms
    max_time = int(350 * 10)  # ms
    time_bin_size = int(params['time_bin_size'])  # ms
    global time
    time = range(0, max_time, time_bin_size)
    # Increase max_time to make sure no timestep is cut due to fractional amount of steps
    global time_step
    time_step = time_bin_size * 0.001
    data_steps = len(time)

    """
    infile = open(file_name, 'rb')
    data_dict = pickle.load(infile)
    infile.close()
    # Extract data
    data = []
    labels = []
    bins = 1000  # [ms] 
    nchan = len(data_dict[1]['events']) # number of channels/sensors
    """
    dataset = pd.read_pickle(file_name)
    data_dict = dataset.copy()
    data = []
    labels = []
    bins = 1000  # [ms]
    nchan = len(data_dict['events'][1])  # number of channels/sensors
    for i, sample in enumerate(data_dict['events']):
        dat = (sample[:])
        events_array = np.zeros([nchan, round((max_time / time_bin_size) + 0.5), 2])
        for taxel in range(len(dat)):
            for event_type in range(len(dat[taxel])):
                if dat[taxel][event_type]:
                    indx = bins * (np.array(dat[taxel][event_type]))
                    indx = np.array((indx / time_bin_size).round(), dtype=int)
                    events_array[taxel, indx, event_type] = 1
        if taxels != None:
            events_array = np.reshape(np.transpose(events_array, (1, 0, 2))[:, taxels, :], (events_array.shape[1], -1))
            selected_chans = 2 * len(taxels)
        else:
            events_array = np.reshape(np.transpose(events_array, (1, 0, 2)), (events_array.shape[1], -1))
            selected_chans = 2 * nchan
        data.append(events_array)
        labels.append(letter_written.index(data_dict['letter'][i]))

    data = torch.tensor(data, dtype=dtype)
    labels = torch.tensor(labels, dtype=torch.long)

    train, val, test = ratios

    if val > 0:
        split_1 = 1 - train
        split_2 = 1 - val / (val + test)
        x_train, x_valtest, y_train, y_valtest = train_test_split(data, labels, test_size=split_1, shuffle=True,
                                                                  stratify=labels, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_valtest, y_valtest, test_size=split_2, shuffle=True,
                                                        stratify=y_valtest, random_state=42)
        ds_train = TensorDataset(x_train, y_train)
        ds_val = TensorDataset(x_val, y_val)
        ds_test = TensorDataset(x_test, y_test)
    else:
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test, shuffle=True, stratify=labels,
                                                            random_state=42)
        ds_train = TensorDataset(x_train, y_train)
        ds_val = []
        ds_test = TensorDataset(x_test, y_test)

    return ds_train, ds_val, ds_test, labels, selected_chans, data_steps

def load_data(file_name="./data/data_braille_letters_0.0.pkl", upsample_fac=1.0, norm_val=1, filtering=False, specify_letters = []):
    '''
    Load the tactile Braille data.
    '''
    # TODO inlcude a dataloader (to open loading other data)
    data_dict = pd.read_pickle(file_name)  # 1kHz data (new dataset)

    # Extract data
    if len(specify_letters) != 0:
        letter_list = []
        for letter in specify_letters:
            letter_list.append(np.where(data_dict['letter'] == letter)[0])
            labels = data_dict['letter'][np.array(letter_list).flatten()]
            data = data_dict['taxel_data'][np.array(letter_list).flatten()]
    else:
        data = data_dict['taxel_data']
        labels = data_dict['letter']
    timestamps = data_dict['timestamp']
    # TODO find a way to use letters as labels
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)  # labels as int numbers
    data_steps = len(data[0])

    # filter and resample
    data_resampled = []
    data_resampled_split = []
    timestamps_resampled = []
    filter_size = [int((data_steps*upsample_fac)/75), 0]  # found manually
    max_val = 111.0/norm_val  # found by iterating over data beforehand
    for counter, trial in enumerate(data):
        # no resampling, just filtering (smoothing)
        if upsample_fac == 1.0:
            data_dummy = trial
            timestamps_dummy = timestamps[counter]
            if  filtering:
                data_dummy = ndimage.uniform_filter(
                    trial, size=filter_size, mode='nearest')  # smooth

        # downsampling and filtering
        elif upsample_fac < 1.0:
            data_dummy = signal.decimate(trial, int(
                1/upsample_fac), axis=0)  # downsample
            time_interpolate = interp1d(range(data_steps), timestamps[counter])
            timestamps_dummy = time_interpolate(np.linspace(0, data_steps-1, int(data_steps/int(1/upsample_fac)+0.5)))
            if filtering:
                data_dummy = ndimage.uniform_filter(
                    data_dummy, size=filter_size, mode='nearest')  # smooth
        
        # upsampling and filtering
        else:
            data_dummy = signal.resample(
                trial, int(data_steps*upsample_fac))  # upsample
            time_interpolate = interp1d(range(data_steps), timestamps[counter])
            timestamps_dummy = time_interpolate(np.linspace(0, data_steps-1, int(data_steps*upsample_fac)))
            if filtering:
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
        timestamps_resampled.append(timestamps_dummy[int(data_steps*0.1):]-timestamps_dummy[int(data_steps*0.1)])
        data_resampled.append(data_dummy[int(data_steps*0.1):])
        data_resampled_split.append(data_split[int(data_steps*0.1):])

    data_steps = len(data_resampled[0])  # update data steps

    data = torch.as_tensor(np.array(data_resampled), dtype=torch.float) 
    data_split = torch.as_tensor(np.array(data_resampled_split), dtype=torch.float)  # data to feed to encoding
    labels = torch.as_tensor(labels, dtype=torch.long)

    # selected_chans = 2*len(data[0][0])
    return data_split, labels, timestamps_resampled, data_steps, le, data

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
