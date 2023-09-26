import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
import h5py


def set_random_seed(seed, add_generator=False, device=torch.device('cpu')):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if add_generator:
        generator = torch.Generator(device=device).manual_seed(seed)
        return generator
    else:
        return None


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
        print(self.n_time_steps)

    def __call__(self, sample):
        # Map 2D input image to 2D tensor with px ids and current:

        sample = sample.flatten(start_dim=1, end_dim=2).unsqueeze(1).repeat(1, self.n_time_steps, 1)
        if self.add_noise:
            # TODO: Check how to add noise
            sample = sample.to(torch.float) + torch.randint_like(sample, high=1) * self.v_max

        return sample


def load_MNIST(batch_size=1, stim_len_sec=1, dt_sec=1e-3, v_max=0.2, generator=None,
               n_samples_train=None, n_samples_test=None, subset_classes=None):
    """
    Load MNIST dataset and return train and test loader.

    :param n_samples_test: if not None, select a subset of samples of size n_samples_test for the test set
    :param n_samples_train: if not None, select a subset of samples of size n_samples_train for the train set
    :param batch_size: batch size
    :param threshold_grayscale: threshold of ToBin transformation
    :param vmin: min value of binarized pixels
    :param vmax: max value of binarized pixels
    :param stim_len_sec: length Poisson process (in sec)
    :param dt_sec: dt simulation
    :param r_max: max rate of ToPoisson transformation
    :param r_min: min rate of ToPoisson transformation
    :param generator: generator object of DataLoader
    :param subset_classes: if not empty, select only samples of the specified class

    :return: dictionary with train and test loader and dataset attributes
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: num_workers > 0 and pin_memory True does not work on pytorch 1.12
    # try pytorch 1.13 with CUDA > 1.3
    kwargs = {'num_workers': 4, 'pin_memory': True}
    # kwargs = {}

    # Train:
    trainset = MNIST(root='data', train=True, download=True,
                     transform=transforms.Compose([transforms.PILToTensor(),
                                                   ToCurrent(stim_len_sec, dt_sec, v_max)]))

    trainset = extract_samples(trainset, n_samples_train, subset_classes)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              generator=generator,
                              **kwargs)
    print(f'N samples training: {len(trainset.data)}')

    # Test:
    testset = MNIST(root='data', train=False, download=True,
                    transform=transforms.Compose([transforms.PILToTensor(),
                                                  ToCurrent(stim_len_sec, dt_sec, v_max)]))
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


# TODO: Remove this after testing class below
class EventDataset(Dataset):
    """
    Load MNIST dataset as EventBased dataset generated by converting MNIST digits into Poisson spike train (using ToBin
    and ToPoisson transformations).
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
        spike_times = self.file['spike_times'][idx]
        neuron_ids = self.file['neuron_ids'][idx]
        class_label = self.file['class_labels'][idx]
        spike_times_teacher = self.file['spikes_teacher'][idx]
        poisson_rates = self.file['poisson_rates'][idx]
        spike_times_inh_per_neuron = self.file['spike_times_inh'][idx]

        # Exc input
        i = np.array([spike_times, neuron_ids])
        v = np.ones(len(neuron_ids), dtype=int)
        sparse_spikes_exc = torch.sparse_coo_tensor(i, v, (self.n_time_steps,
                                                           self.n_inputs))

        # Inh input:
        neuron_ids = np.concatenate([[i] * len(spk_time) for i, spk_time in enumerate(spike_times_inh_per_neuron)])
        spike_times = np.concatenate(spike_times_inh_per_neuron)
        sparse_spikes_inh = torch.sparse_coo_tensor(np.array([spike_times, neuron_ids]),
                                                    np.ones(len(neuron_ids), dtype=int),
                                                    (self.n_time_steps, self.n_classes))
        input_spikes_inh = sparse_spikes_inh.to_dense().type(torch.FloatTensor).to(self.device)

        # Teacher:
        spikes_teacher = torch.zeros(self.n_time_steps)
        spikes_teacher[spike_times_teacher] = 1
        spikes_teacher = spikes_teacher.to(self.device)

        return [sparse_spikes_exc.to_dense(), poisson_rates], class_label, spikes_teacher, input_spikes_inh

    def __len__(self):
        return len(self.file['class_labels'])


class EventDataset_Chunks(Dataset):
    """
    Load MNIST dataset as EventBased dataset generated by converting MNIST digits into Poisson spike train (using ToBin
    and ToPoisson transformations).
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
        spike_times_exc = self.file['spike_times_exc'][idx]
        neuron_ids_exc = self.file['neuron_ids_exc'][idx]
        class_label = self.file['class_labels'][idx]
        spike_times_teacher = self.file['spike_times_teacher'][idx]
        neuron_ids_teacher = self.file['neuron_ids_teacher'][idx]
        poisson_rates = self.file['poisson_rates'][idx]
        spike_times_inh = self.file['spike_times_inh'][idx]
        neuron_ids_inh = self.file['neuron_ids_inh'][idx]
        n_output_neurons = self.n_classes * self.n_neurons_per_class

        # Exc input:
        i = np.array([spike_times_exc, neuron_ids_exc])
        v = np.ones(len(neuron_ids_exc), dtype=int)
        spikes_exc = torch.sparse_coo_tensor(i, v, (self.n_time_steps, self.n_inputs)).to_dense()
        spikes_exc = spikes_exc.type(torch.bool).to(self.device)

        # Inh input:
        i = np.array([spike_times_inh, neuron_ids_inh])
        v = np.ones(len(neuron_ids_inh), dtype=int)
        spikes_inh = torch.sparse_coo_tensor(i, v, (self.n_time_steps, n_output_neurons)).to_dense().type(
            torch.FloatTensor).to(self.device)
        spikes_inh = spikes_inh.type(torch.bool).to(self.device)

        # Teacher input:
        i = np.array([spike_times_teacher, neuron_ids_teacher])
        v = np.ones(len(neuron_ids_teacher), dtype=int)
        spikes_teacher = torch.sparse_coo_tensor(i, v, (self.n_time_steps, n_output_neurons)).to_dense().type(
            torch.FloatTensor).to(self.device)
        spikes_teacher = spikes_teacher.type(torch.bool).to(self.device)

        return [spikes_exc, poisson_rates], class_label, spikes_teacher, spikes_inh

    def __len__(self):
        return len(self.file['class_labels'])
