import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from scipy import signal
from numpy.fft import rfft, rfftfreq


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
    frange = np.linspace(center - span / 2, center + span / 2, samples_n)
    data_f = torch.zeros(data.shape[0], samples_n)

    assert (len(data.shape) == 3)  # batch size x freq range x n channels

    for f in range(len(frange) - 1):
        idx = np.where(np.logical_and(freqs >= frange[f], freqs < frange[f + 1]))[0]
        data_f[:, f] = torch.mean(data[:, idx])  # mean across freq range and channels

    return data_f


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
        x = signal.filtfilt(b, a, x, axis=0)
    n_samples = x.shape[0]
    xf = rfftfreq(n_samples, dt)
    yf = rfft(x, axis=0)
    yf = np.abs(yf)
    if average:
        yf = np.mean(yf, axis=1)
    if normalize:
        yf = yf / np.max(yf)

    return xf, yf


class ToCurrent(object):
    """
    Custom data transformation.

    Map pixel values to current amplitude across time. 

    """

    def __init__(self, stim_len_sec, dt_sec=1e-2, v_max=0.2, add_noise=True):
        """
        :param sitm_length_sec: stimulus duration (in sec)
        :param dt_sec: stimulus dt (in sec)
        """

        self.stim_len_sec = stim_len_sec
        self.dt_sec = dt_sec
        self.n_time_steps = int(self.stim_len_sec / self.dt_sec)
        self.add_noise = add_noise
        self.v_max = v_max

    def __call__(self, sample):
        # Map 2D input image to 2D tensor with px ids and current:

        sample = sample.flatten(start_dim=1, end_dim=2).unsqueeze(1).repeat(1, self.n_time_steps, 1)
        if self.add_noise:
            # TODO: Check how to add noise
            sample = sample.to(torch.float) + torch.randint_like(sample, high=10) / 10 * self.v_max

        # Return tensor without channel dimension (only grayscale samples)
        return sample[0]


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


def load_MNIST(batch_size=1, stim_len_sec=1, dt_sec=1e-3, v_max=0.2, generator=None,
               n_samples_train=None, n_samples_test=None, subset_classes=None, add_noise=True, return_fft=False):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: num_workers > 0 and pin_memory True does not work on pytorch 1.12
    # try pytorch 1.13 with CUDA > 1.3
    kwargs = {'num_workers': 4, 'pin_memory': True}
    # kwargs = {}

    if return_fft:
        # Return data in frequency domain:
        list_transforms = [transforms.PILToTensor(),
                           ToCurrent(stim_len_sec, dt_sec, v_max, add_noise=add_noise),
                           ToFft(dt_sec)]
    else:
        # Return samples in time:
        list_transforms = [transforms.PILToTensor(),
                           ToCurrent(stim_len_sec, dt_sec, v_max, add_noise=add_noise)]
    # Train:
    trainset = MNIST(root='data', train=True, download=True,
                     transform=transforms.Compose(list_transforms))
    trainset = extract_samples(trainset, n_samples_train, subset_classes)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              generator=generator,
                              **kwargs)
    print(f'N samples training: {len(trainset.data)}')

    # Test:
    testset = MNIST(root='data', train=False, download=True,
                    transform=transforms.Compose(list_transforms))
    testset = extract_samples(testset, n_samples_test, subset_classes)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             shuffle=True,
                             generator=generator,
                             **kwargs)
    print(f'N samples test: {len(testset.data)}')

    return train_loader, test_loader


def load_dataset(dataset_name, **kwargs):
    if dataset_name == 'MNIST':
        train_loader, test_loader = load_MNIST(**kwargs)
    else:
        pass

    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    batch_size, n_time_steps, n_inputs = example_data[0].shape
    n_classes = len(train_loader.dataset.classes)
    dt = kwargs['dt_sec']

    return {'train_loader': train_loader,
            'test_loader': test_loader,
            'batch_size': batch_size,
            'n_time_steps': n_time_steps,
            'n_inputs': n_inputs,
            'n_classes': n_classes,
            'dt_sec': dt}


def extract_samples(set, n_samples, subset_classes):
    if n_samples is not None:
        print('Extracting a subset of samples')
        set.data = set.data[range(n_samples)]
        set.targets = set.targets[range(n_samples)]
    if subset_classes is not None:
        print('Extracting a subset of classes')
        idx_to_extract = []
        for c in subset_classes:
            idx_to_extract.extend(torch.where(set.targets == c)[0])
        set.data = set.data[np.array(idx_to_extract)]
        set.targets = set.targets[np.array(idx_to_extract)]

    assert len(set.data) == len(set.targets)

    return set
