import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
from scipy import signal
from numpy.fft import rfft, rfftfreq
from datasets import load_data
from sklearn.model_selection import train_test_split
import torch.nn as nn

def set_random_seed(seed, add_generator=False, device=torch.device('cpu')):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if add_generator:
        generator = torch.Generator(device=device).manual_seed(seed)
        return generator
    else:
        return None


def extract_interval(data, freqs, samples_n, center, span):
    """
    Map data to mean of fft centered on center with range = span
    """
    frange = np.linspace(center - span / 2, center + span / 2, samples_n)
    data_f = torch.zeros(data.shape[0], samples_n)

    assert (len(data.shape) == 3)  # batch size x freq range x n channels

    for f in range(len(frange) - 1):
        idx = np.where(np.logical_and(freqs >= frange[f], freqs < frange[f + 1]))[0]
        if np.isnan(np.array(torch.mean(data[:, idx]))):
            data_f[:, f] = 0
        else:
            data_f[:, f] = torch.mean(data[:, idx])  # mean across freq range and channels

    return data_f


def extract_histogram(data, bins_hist, center, span):
    """
    Map data to histgram of values centered on center with range = span
    """
    bins_coll = []
    for trial in range(data.shape[0]):
        datax = data[trial].flatten()
        bins, edges = np.histogram(datax, bins=bins_hist, range=(center - span / 2,
                                                                 center + span / 2))

        bins_coll.append(bins)
    data_hist = np.array(bins_coll)

    return data_hist


def get_fft(data, dt, high_pass=True, normalize=True, average=True):
    """
    Apply fft to input data
    :param data: time bins x channels
    :param dt inverse of samplinf freq
    :param high_pass, if True, a high pass filter is applied on the data
    """
    x = data
    if high_pass:
        b, a = signal.butter(3, 0.1, 'high')
        x = signal.filtfilt(b, a, x, axis=1)
    n_samples = x.shape[0]
    xf = rfftfreq(n_samples, dt)
    yf = rfft(x, axis=1)
    yf = np.abs(yf)
    if average:
        yf = np.mean(yf, axis=2)
    if normalize:
        yf = yf / np.max(yf)

    return xf, yf


class ToCurrent(object):
    """
    Custom data transformation.

    Map pixel values to current amplitude across time. 

    """

    def __init__(self, stim_len_sec, dt_sec=1e-2, v_max=0.2, add_noise=True,gain = 1.):
        """
        :param sitm_length_sec: stimulus duration (in sec)
        :param dt_sec: stimulus dt (in sec)
        """

        self.stim_len_sec = stim_len_sec
        self.dt_sec = dt_sec
        self.n_time_steps = int(self.stim_len_sec / self.dt_sec)
        self.add_noise = add_noise
        self.v_max = v_max
        self.gain = gain

    def __call__(self, sample):
        # Map 2D input image to 2D tensor with px ids and current:

        if len(sample.shape) == 3:
            # If without ToEnc transformation
            sample = sample.flatten(start_dim=1, end_dim=2).unsqueeze(1).repeat(1, self.n_time_steps, 1)
            if self.add_noise:
                # TODO: Check how to add noise
                sample = (sample.to(torch.float) + torch.randint_like(sample, high=10) / 10 * self.v_max) * self.gain
            else:
                sample *= self.gain
            sample = sample[0]

        elif len(sample.shape) == 1:
            # If after ToEnc transformation
            sample = sample.unsqueeze(0).repeat(self.n_time_steps, 1)
            if self.add_noise:
                # TODO: Check how to add noise
                sample = (sample.to(torch.float) + torch.randint_like(sample, high=10) / 10 * self.v_max) * self.gain
            else:
                sample *= self.gain
        else:
            raise ValueError

        assert sample.shape[0] == self.n_time_steps
        # sample.shape[1] == n features

        return sample

class ToEnc(object):
    """
    Custom data transformation.

    Map pixel values to current amplitude across time.

    """

    def __init__(self, encoder_model, encoding_dim=6):
        """
        :param encoder_model: '.pt with model autoencoder'
        :param dt_sec: stimulus dt (in sec)
        """
        self.model = Autoencoder_linear(encoding_dim)
        self.model.load_state_dict(torch.load(encoder_model))
        print(self.model)

    def __call__(self, sample):
        # Map 2D input image to 2D tensor with px ids and current:

        sample_flatten = sample[0].flatten()

        # get sample outputs
        output = self.model(sample_flatten.float())
        encs = self.model.encoder(sample_flatten.float()).clone().detach()

        return encs

class ToFft(object):
    """
    Custom data transformation.

    Map current values to frequency spectrum.

    """

    def __init__(self, dt_sec=1e-2):
        self.dt_sec = dt_sec

    def __call__(self, sample):
        # Get fft:
        _, yf = get_fft(sample, self.dt_sec, high_pass=True, normalize=False, average=False)

        return yf

def create_empty_dataset(h5py_file, list_dataset_names, shape, chunks, dtype):
    """
    Initialize input file with empty dataset.
    """
    for field in list_dataset_names:
        h5py_file.create_dataset(field, shape=shape, chunks=chunks, dtype=dtype)
def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    # datasets = {}
    train_val = Subset(dataset, train_idx)
    val_val = Subset(dataset, val_idx)
    return train_val, val_val
def load_MNIST(batch_size=1, stim_len_sec=1, dt_sec=1e-3, v_max=0.2, generator=None, shuffle=True,
               n_samples_train=-1, n_samples_test=-1, subset_classes=None, add_noise=True, return_fft=False,
               train_val_split=0.2,gain=1, compressed=False, encoder_model=None):
    """
    Load MNIST dataset and return train and test loader.

    :param n_samples_test: if not None, select a subset of samples of size n_samples_test for the test set
    :param n_samples_train: if not None, select a subset of samples of size n_samples_train for the train set
    :param batch_size: batch size
    :param v_max:
    :param stim_len_sec: length Poisson process (in sec)
    :param dt_sec: dt simulation
    :param generator: generator object of DataLoader
    :param subset_classes: if not empty, select only samples of the specified class
    :param add_noise:
    :param return_fft
    """

    # TODO: num_workers > 0 and pin_memory True does not work on pytorch 1.12
    # try pytorch 1.13 with CUDA > 1.3
    kwargs = {'num_workers': 4, 'pin_memory': True}

    if return_fft:
        # Return data in frequency domain:
        list_transforms = [transforms.PILToTensor(),
                           ToCurrent(stim_len_sec, dt_sec, v_max, add_noise=add_noise, gain=gain),
                           ToFft(dt_sec)]
    else:
        if compressed:
            buff_trainset = MNIST(root='data', train=True, download=True)
            # Return samples in time:
            list_transforms = [transforms.ToTensor(),
                               transforms.Normalize((buff_trainset.data.float().mean() / 255,),
                                                    (buff_trainset.data.float().std() / 255,)),
                               ToEnc(encoder_model),
                               ToCurrent(stim_len_sec, dt_sec, v_max, add_noise=add_noise, gain=gain)]
        else:
            # Return samples in time:
            list_transforms = [transforms.PILToTensor(),
                               ToCurrent(stim_len_sec, dt_sec, v_max, add_noise=add_noise, gain=gain)]
    # Train:
    trainset = MNIST(root='data', train=True, download=True,
                     transform=transforms.Compose(list_transforms))

    # Measure number of samples per class:
    n_occ = torch.zeros(10)
    for i in range(10):
        n_occ[i] = len(torch.where(trainset.targets == i)[0])
    # print('N samples per class:', n_occ)

    # Extract subset of samples:
    trainset = extract_samples(trainset, n_samples_train, subset_classes)

    # Measure number of samples per class:
    n_occ = torch.zeros(10)
    for i in range(10):
        n_occ[i] = len(torch.where(trainset.targets == i)[0])
    # print('N samples per class:', n_occ)

    # Create data loader:
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              generator=generator,
                              **kwargs)
    # print(f'N samples training: {len(trainset.data)}')

    # Test:
    testset = MNIST(root='data', train=False, download=True,
                    transform=transforms.Compose(list_transforms))

    # Measure number of samples per class:
    n_occ = torch.zeros(10)
    for i in range(10):
        n_occ[i] = len(torch.where(testset.targets == i)[0])
    # print('N samples per class:', n_occ)

    # Extract subset of samples:
    testset = extract_samples(testset, n_samples_test, subset_classes)

    # Measure number of samples per class:
    n_occ = torch.zeros(10)
    for i in range(10):
        n_occ[i] = len(torch.where(testset.targets == i)[0])
    # print('N samples per class:', n_occ)

    # Create data loader:
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             generator=generator,
                             **kwargs)
    # print(f'N samples test: {len(testset.data)}')
    train_val, val_val = train_val_dataset(trainset, val_split=train_val_split)
    train_val_loader = DataLoader(train_val,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             generator=generator,
                             **kwargs)
    val_loader = DataLoader(val_val,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                generator=generator,
                                **kwargs)
    return train_loader, test_loader, train_val_loader, val_loader


def load_Braille(data_path=None, batch_size=None, generator=None, upsample_fac=1, gain=10, shuffle=True,train_val_split=0.2):
    kwargs = {'num_workers': 4, 'pin_memory': True}

    # file_name = "data/data_braille_letters_all.pkl"
    data, labels, _, _, _, _ = load_data(data_path, upsample_fac)
    data *= gain
    nb_channels = data.shape[-1]

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)
    n_classes = len(np.unique(y_train))

    test_loader = DataLoader(ds_test,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             generator=generator,
                             **kwargs)

    train_loader = DataLoader(ds_train,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              generator=generator,
                              **kwargs)
    train_val, val_val = train_val_dataset(ds_train, val_split=train_val_split)
    train_val_loader = DataLoader(train_val,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             generator=generator,
                             **kwargs)
    val_loader = DataLoader(val_val,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                generator=generator,
                                **kwargs)
    return train_loader, test_loader, train_val_loader, val_loader, n_classes


def load_dataset(dataset_name, **kwargs):
    if dataset_name == 'MNIST':
        train_loader, test_loader,train_val_loader,val_loader = load_MNIST(**kwargs)
        dt = kwargs['dt_sec']
        n_classes = len(train_loader.dataset.classes)
    elif dataset_name == 'Braille':
        dt = (1 / kwargs['sampling_freq_hz']) / kwargs['upsample_fac']
        train_loader, test_loader,train_val_loader,val_loader, n_classes = load_Braille(data_path=kwargs['data_path'],
                                                            batch_size=kwargs['batch_size'],
                                                            generator=kwargs['generator'],
                                                            upsample_fac=kwargs['upsample_fac'],
                                                            shuffle=kwargs['shuffle'],
                                                            gain = kwargs['gain'])
    else:
        raise ValueError(f'Dataset {dataset_name} not supported')

    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    batch_size, n_features, n_inputs = example_data.shape

    if kwargs['return_fft']:
        # Data is the frequency spectrum of the signal
        n_time_steps = int(kwargs['stim_len_sec'] / dt)
        n_freq_steps = n_features
    else:
        # Data is an amplitude across time:
        n_time_steps = n_features
        n_freq_steps = -1
    return {'train_loader': train_loader,
            'test_loader': test_loader,
            'train_val_loader':train_val_loader,
            'val_loader':val_loader,
            'batch_size': batch_size,
            'n_freq_steps': n_freq_steps,
            'n_time_steps': n_time_steps,
            'n_features': n_features,
            'n_classes': n_classes,
            'dt_sec': dt,
            'n_inputs': n_inputs,
            'v_max': kwargs['v_max']}


def extract_samples(set, n_samples, subset_classes):
    if n_samples > 0:
        #
        # print(f'Extracting a subset of {n_samples} samples')
        set.data = set.data[range(n_samples)]
        set.targets = set.targets[range(n_samples)]
    if subset_classes is not None:
        # print('Extracting a subset of classes')
        idx_to_extract = []
        for c in subset_classes:
            idx_to_extract.extend(torch.where(set.targets == c)[0])
        set.data = set.data[np.array(idx_to_extract)]
        set.targets = set.targets[np.array(idx_to_extract)]

    assert len(set.data) == len(set.targets)

    return set


class Dataset_features(Dataset):
    """
    Load dataset as custom dataset generated by converting MNIST digits into input currents or into feature sets
    (frequency values, amplitudes and slopes).
    """

    def __init__(self, hdf5_file, device=None):
        """
            :param data_file: Path to h5 file with dataset.
        """
        self.file = hdf5_file
        self.device = device

        for attr in self.file.attrs.keys():
            eval_str = "self.{}".format(attr) + " = " + str(self.file.attrs[attr])
            exec(eval_str)

    def __getitem__(self, idx):
        values = self.file['values'][idx]
        assert(len(values)>0)
        target = self.file['targets'][idx]

        data = values

        return data, target

    def __len__(self):
        return len(self.file['targets'])


class MNISTDataset_current(Dataset):
    """
    Load MNIST dataset as custom dataset generated by converting MNIST digits into input currents or into feature sets
    (frequency values, amplitudes and slopes).
    """

    def __init__(self, hdf5_file, device=None):
        """
            :param data_file: Path to h5 file with dataset.
        """
        self.file = hdf5_file
        self.device = device

        for attr in self.file.attrs.keys():
            eval_str = "self.{}".format(attr) + " = " + str(self.file.attrs[attr])
            exec(eval_str)

    def __getitem__(self, idx):
        values = self.file['values'][idx]
        target = self.file['targets'][idx]
        idx_time = self.file['idx_time'][idx]
        idx_inputs = self.file['idx_inputs'][idx]
        idx = np.vstack((idx_time, idx_inputs))
        # From sparse to dense
        data = torch.sparse_coo_tensor(idx, values, (self.n_time_steps, self.n_inputs)).to_dense()

        return data, target

    def __len__(self):
        return len(self.file['targets'])

class Autoencoder_linear(nn.Module):
    def __init__(self, encoding_dim, input_dim=784, output_dim=10):
        super(Autoencoder_linear, self).__init__()
        ## encoder ##
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim*4),
            nn.ReLU(),
            nn.Linear(encoding_dim*4, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim),
            nn.ReLU()
        )

        ## decoder ##
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, output_dim)
        )

    def forward(self, x):
        # define feedforward behavior
        # and scale the *output* layer with a sigmoid activation function
        x = x.float()
        # pass x into encoder
        out = self.encoder(x)
        # pass out into decoder
        out = self.decoder(out)

        return out