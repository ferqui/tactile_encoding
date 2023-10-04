import matplotlib.pyplot as plt

from utils_dataset import set_random_seed, MNISTDataset_features, MNISTDataset_current
from torch import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import torch
from tqdm import trange
import argparse
import h5py


def main(args):

    # Device: ----------------------------------------------------------------------------------------------------------
    device = torch.device(args.device)
    if device == torch.device("cuda") and not (torch.cuda.is_available()):
        device == torch.device('cpu')
        print('Could not find CUDA. Running on CPU')
    print(f'Running on {device}')

    if device == torch.device("cuda"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.empty_cache()
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Folders: ---------------------------------------------------------------------------------------------------------
    folder = Path('dataset_analysis')
    folder_run = Path(os.path.join(folder, 'MNIST'))
    folder_fig = folder_run.joinpath('fig')
    folder_run.mkdir(parents=True, exist_ok=True)
    folder_fig.mkdir(parents=True, exist_ok=True)
    folder_data = folder_run.joinpath('data')
    folder_data.mkdir(parents=True, exist_ok=True)

    # Dataseet: --------------------------------------------------------------------------------------------------------
    generator = set_random_seed(args.seed, add_generator=True, device=device)

    if args.n_samples_train == -1:
        # use all training samples from MNIST
        trainset = MNIST(root='data', train=True, download=False)
        n_samples_train = len(trainset)
    else:
        n_samples_train = args.n_samples_train
    if args.n_samples_test == -1:
        testset = MNIST(root='data', train=False, download=False)
        n_samples_test = len(testset)
    else:
        n_samples_test = args.n_samples_test

    path_to_dataset = Path('dataset')
    path_to_dataset.mkdir(parents=True, exist_ok=True)
    dict_dataset = {}
    if args.data_type == 'current':
        filename_dataset = path_to_dataset.joinpath('MNIST',
                                                    args.data_type,
                                                    f'{n_samples_train}_{n_samples_test}')
        train_dataset = MNISTDataset_current(h5py.File(filename_dataset.joinpath('train.h5'), 'r'), device=device)
        test_dataset = MNISTDataset_current(h5py.File(filename_dataset.joinpath('test.h5'), 'r'), device=device)

    else:
        filename_dataset = path_to_dataset.joinpath('MNIST',
                                                    args.data_type,
                                                    f'{n_samples_train}_{n_samples_test}_c{args.center}_s{args.span}')
        train_dataset = MNISTDataset_features(h5py.File(filename_dataset.joinpath('train.h5'), 'r'), device=device)
        test_dataset = MNISTDataset_features(h5py.File(filename_dataset.joinpath('test.h5'), 'r'), device=device)

    dict_dataset['train_loader'] = DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              generator=generator)
    dict_dataset['test_loader'] = DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             generator=generator)

    for i, input_data in enumerate(tqdm(dict_dataset['train_loader'])):
        data = input_data[0]
        targets = input_data[1]

        # Check dimensions:
        assert data.shape[0] == args.batch_size

        if args.data_type == 'current':
            assert len(data.shape) == 3
            assert data.shape[1] == train_dataset.n_time_steps
            assert data.shape[2] == train_dataset.n_inputs
            for j in range(10):
                # Save output samples:
                np.save(filename_dataset.joinpath(f's{j}_t{targets[j][0]}.npy'), data[j])
        else:
            assert len(data.shape) == 2
            assert data.shape[1] == train_dataset.sample_size

        assert targets.shape[0] == args.batch_size
        assert targets.shape[1] == 1

        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser('dataloader')

    parser.add_argument('--seed',
                        type=int,
                        default=10)
    parser.add_argument('--n_samples_train',
                        type=int,
                        default=500)
    parser.add_argument('--n_samples_test',
                        type=int,
                        default=10)
    parser.add_argument('--data_type',
                        type=str,
                        default='current',
                        choices=['current', 'frequency', 'amplitude', 'slope'])
    parser.add_argument('--center',
                        type=int,
                        default=10)
    parser.add_argument('--span',
                        type=int,
                        default=10)
    parser.add_argument('--sample_size',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        type=int,
                        default=100)
    parser.add_argument('--device',
                        type=str,
                        choices=['cpu', 'cuda'],
                        default='cpu')

    args = parser.parse_args()

    # Run:
    main(args)
