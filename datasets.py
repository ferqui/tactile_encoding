import torch
import pickle
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

def upsample(data, n=2):
    shp = data.shape
    tmp = data.reshape(shp + (1,))
    tmp = data.tile((1, 1, 1, n))
    return tmp.reshape((shp[0], n * shp[1], shp[2]))

def load_analog_data(params):
    # data structure: [trial number] x ['key'] x [time] x [sensor_nr]
    import gzip
    file_name = 'data/tutorial5_braille_spiking_data.pkl.gz'
    with gzip.open(file_name, 'rb') as infile:
        data_dict = pickle.load(infile)

    max_time = int(54*25) #ms TODO: Why?
    time_bin_size = int(params['time_bin_size']) # ms
    time = range(0,max_time,time_bin_size)


    letter_written = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                      'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # nb_channels = data_dict[0]['taxel_data'].shape[1]
    nb_channels = 12 #We did it because Zenke takes 4 sensors
    # Extract data
    nb_repetitions = 50

    data = []
    labels = []

    # data_dict = interpolate_data(data_dict,interpolate_size=500)
    for i, letter in enumerate(letter_written):
        for repetition in np.arange(nb_repetitions):
            idx = i * nb_repetitions + repetition
            dat = 1.0 - data_dict[idx]['taxel_data'][:] / 255
            data.append(dat)
            labels.append(i)

    # Crop to same length
    data_steps = l = np.min([len(d) for d in data])
    data = torch.tensor(np.array([d[:l] for d in data]), dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    file_name = 'data/data_tactile_processes.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

    file_name = 'data/labels_tactile_processes.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


    # Standardize data
    rshp = data.reshape((-1, data.shape[2]))
    data = (data - rshp.mean(0)) / (rshp.std(0) + 1e-3)

    nb_upsample = 2 # TODO: Why 2?
    data = upsample(data, n=nb_upsample)

    # Shuffle data
    idx = np.arange(len(data))
    np.random.seed(0)
    np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]

    # Peform train/test split
    a = int(0.8 * len(idx))
    x_train, x_test = data[:a], data[a:]
    y_train, y_test = labels[:a], labels[a:]

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)

    return ds_train, ds_test, labels, nb_channels, data_steps*nb_upsample