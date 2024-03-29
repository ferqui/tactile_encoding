import numpy as np
import torch
import torchvision as tv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms
import torch.nn as nn


class ToCurrent(object):
    """
    Custom data transformation.

    Map pixel values to current amplitude across time.

    """

    def __init__(self, stim_len_sec, dt_sec=1e-2, v_max=0.2, add_noise=True, gain=1.0):
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
            sample = (
                sample.flatten(start_dim=1, end_dim=2)
                .unsqueeze(1)
                .repeat(1, self.n_time_steps, 1)
            )
            if self.add_noise:
                # TODO: Check how to add noise
                # TODO: Check if this is the right way to add noise
                # TODO: Check if this is the left way to add noise
                sample = (
                    sample.to(torch.float)
                    + torch.randint_like(sample, high=10) / 10 * self.v_max
                ) * self.gain
            else:
                sample *= self.gain
            sample = sample[0]

        elif len(sample.shape) == 1:
            # If after ToEnc transformation
            sample = sample.unsqueeze(0).repeat(self.n_time_steps, 1)
            if self.add_noise:
                # TODO: Check how to add noise
                sample = (
                    sample.to(torch.float)
                    + torch.randint_like(sample, high=10) / 10 * self.v_max
                ) * self.gain
            else:
                sample *= self.gain
        else:
            raise ValueError

        assert sample.shape[0] == self.n_time_steps

        return sample
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
class MNISTDataset:
    def __init__(
        self,
        num_train,
        num_test,
        val_size=0.2,
        batch_size=32,
        stim_len_sec=3,
        dt_sec=1e-2,
        v_max=0.2,
        add_noise=True,
        gain=1.0,
        compressed=False,
        encoder_model=None
    ):
        num_val = int(num_train * val_size)
        num_train -= num_val

        self.num_train = num_train
        self.num_test = num_test
        self.num_val = num_val
        self.batch_size = batch_size
        self.compressed = compressed
        self.n_time_steps = int(stim_len_sec / dt_sec)
        self.dt_sec = dt_sec



        if self.compressed:
            transform = tv.transforms.Compose(
                [
                    transforms.PILToTensor(),ToEnc(encoder_model),
                    ToCurrent(stim_len_sec, dt_sec, v_max, add_noise=add_noise, gain=gain),
                ]
            )
        else:
            transform = tv.transforms.Compose(
                [
                    transforms.PILToTensor(),
                    ToCurrent(stim_len_sec, dt_sec, v_max, add_noise=add_noise, gain=gain),
                ]
            )

        train_val_dataset = tv.datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        test_dataset = tv.datasets.MNIST(
            "./data", train=False, download=True, transform=transform
        )
        if num_val > 0:
            train_indices, val_indices = train_test_split(
                range(len(train_val_dataset)),
                test_size=num_val,
                random_state=0,
                shuffle=True,
            )
        else:
            train_indices = range(len(train_val_dataset))
            val_indices = []
        test_indices = np.random.permutation(np.arange(len(test_dataset)))[:-num_test]

        # to help with trying to do neuroevolution since the full dataset is a bit much for evolving convnets...
        train_indices = train_indices[:num_train]
        val_indices = val_indices[: int(len(val_indices))]

        ########################################################
        train_split = Subset(train_val_dataset, train_indices)
        train_dl = iter(DataLoader(train_split, batch_size=num_train, shuffle=False))
        # examples = enumerate(train_dl)
        # batch_idx, (example_data, example_targets) = next(examples)

        self.x_train, self.y_train = next(train_dl)
        ########################################################
        if num_val > 0:
            val_split = Subset(train_val_dataset, val_indices)
            val_dl = iter(DataLoader(val_split, batch_size=num_val, shuffle=False))
            self.x_val, self.y_val = next(val_dl)
        ########################################################
        test_split = Subset(test_dataset, test_indices)
        test_dl = iter(DataLoader(test_split, batch_size=num_test, shuffle=False))
        self.x_test, self.y_test = next(test_dl)


    @property
    def n_inputs(self):
        return self.x_test.shape[-1]
    @property
    def n_features(self):
        return self.x_test.shape[-2]
    @property
    def n_classes(self):
        return len(torch.unique(torch.flatten(self.y_test)))
    def get_train_dataloader(self,device):
        cutoff = self.num_train - self.num_train % self.batch_size

        permutation = torch.randperm(self.num_train)[:cutoff]
        obs = self.x_train[permutation]
        labels = self.y_train[permutation]

        # obs = obs.reshape((-1, self.batch_size) + obs.shape[1:]).to(device)
        obs = obs.to(device)
        labels = labels.to(device)
        return DataLoader(
            TensorDataset(obs, labels),
            batch_size=self.batch_size,
            shuffle=False,
        )
    def get_test_dataloader(self,device):
        cutoff = self.num_test - self.num_test % self.batch_size

        obs = self.x_test[:cutoff]
        labels = self.y_test[:cutoff]

        obs = obs.to(device)
        labels = labels.to(device)
        return DataLoader(
            TensorDataset(obs, labels),
            batch_size=self.batch_size,
            shuffle=False,
        )
    def get_train(self, device="cuda"):
        cutoff = self.num_train - self.num_train % self.batch_size

        permutation = torch.randperm(self.num_train)[:cutoff]
        obs = self.x_train[permutation]
        labels = self.y_train[permutation]

        obs = obs.reshape((-1, self.batch_size) + obs.shape[1:]).to(device)
        labels = labels.reshape((-1, self.batch_size)).to(device)

        return obs, labels

    def get_test(self, device="cuda"):
        cutoff = self.num_test - self.num_test % self.batch_size

        obs = self.x_test[:cutoff]
        labels = self.y_test[:cutoff]

        obs = obs.reshape((-1, self.batch_size) + obs.shape[1:]).to(device)
        labels = labels.reshape((-1, self.batch_size)).to(device)

        return obs, labels

    def get_val(self, device="cuda"):
        cutoff = self.num_val - self.num_val % self.batch_size

        obs = self.x_val[:cutoff]
        labels = self.y_val[:cutoff]

        obs = obs.reshape((-1, self.batch_size) + obs.shape[1:]).to(device)
        labels = labels.reshape((-1, self.batch_size)).to(device)

        return obs, labels
