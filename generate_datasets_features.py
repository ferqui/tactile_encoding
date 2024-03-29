"""
Generate .h5 file with samples from MNIST dataset saved as chunks with size one. This allows to deal with unlimited size
datasets as the output dataset is saved and updated at every chunk (of size 1), thus without requiring to accumulate
a list of samples before saving the full dataset at once.

Transform input MNIST samples to current across time (with noise).

- N.Risi
"""

import torch
import argparse
from utils_dataset import load_dataset, set_random_seed, extract_interval, extract_histogram, create_empty_dataset
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
import os
from numpy.fft import rfft, rfftfreq
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader as Dataloader

def main(args):
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    generator = set_random_seed(args.seed, add_generator=True, device=device)

    # Load dataset:
    if args.dataset == 'Braille':
        dict_dataset = load_dataset('Braille',
                                    batch_size=args.batch_size,
                                    generator=generator,
                                    upsample_fac=1,
                                    data_path="data/data_braille_letters_all.pkl",
                                    return_fft=False,
                                    sampling_freq_hz=100.0,
                                    v_max=-1,
                                    shuffle=True)
    elif args.dataset == 'MNIST':
        dict_dataset = load_dataset('MNIST',
                                    batch_size=args.batch_size,
                                    stim_len_sec=args.stim_len_sec,
                                    dt_sec=args.dt_sec,
                                    v_max=args.v_max,
                                    generator=generator,
                                    add_noise=True,
                                    return_fft=args.data_type == 'frequency',
                                    n_samples_train=args.n_samples_train,
                                    n_samples_test=args.n_samples_test,
                                    shuffle=True)
    elif args.dataset == 'MNIST_compressed':
        dict_dataset = load_dataset('MNIST',
                                    batch_size=args.batch_size,
                                    stim_len_sec=args.stim_len_sec,
                                    dt_sec=args.dt_sec,
                                    v_max=args.v_max,
                                    generator=generator,
                                    add_noise=True,
                                    return_fft=False,
                                    n_samples_train=args.n_samples_train,
                                    n_samples_test=args.n_samples_test,
                                    shuffle=True,
                                    compressed=True,
                                    encoder_model='./data/784MNIST_2_6MNIST.pt')
    else:
        raise NotImplementedError
    path_to_dataset = Path(args.home_dataset)
    path_to_dataset.mkdir(parents=True, exist_ok=True)

    if args.data_type == 'current':
        filename_dataset = path_to_dataset.joinpath(args.dataset,
                                                    args.data_type,
                                                    str(len(dict_dataset['train_loader'].dataset)) + '_' +
                                                    str(len(dict_dataset['test_loader'].dataset)))
    else:
        # Name dataset including info about center and span:
        filename_dataset = path_to_dataset.joinpath(args.dataset,
                                                    args.data_type,
                                                    str(len(dict_dataset['train_loader'].dataset)) + '_' +
                                                    str(len(dict_dataset['test_loader'].dataset)) + '_c' +
                                                    str(args.center) + '_s' +
                                                    str(args.span))

    output_path = Path(filename_dataset)
    output_path.mkdir(parents=True, exist_ok=True)
    val_type = h5py.vlen_dtype(np.dtype('float32'))
    ids_type = h5py.vlen_dtype(np.dtype('int32'))

    for subset in ['train', 'test','train_val','val']:

        n_tot_samples = len(dict_dataset[subset + '_loader'].dataset)
        print(f'N samples {subset}: {n_tot_samples}')

        output_filename = str(filename_dataset.joinpath(subset + '.h5'))
        with h5py.File(output_filename, 'w') as out:
            out.attrs['n_time_steps'] = dict_dataset['n_time_steps']
            out.attrs['n_freq_steps'] = dict_dataset['n_freq_steps']
            out.attrs['n_features'] = dict_dataset['n_features']
            out.attrs['n_classes'] = dict_dataset['n_classes']
            out.attrs['n_inputs'] = dict_dataset['n_inputs']
            out.attrs['dt_sec'] = dict_dataset['dt_sec']
            out.attrs['v_max'] = dict_dataset['v_max']
            out.attrs['batch_size'] = dict_dataset['batch_size']
            out.attrs['center'] = args.center
            out.attrs['span'] = args.span
            create_empty_dataset(out, ['values', 'idx_time', 'idx_inputs', 'targets'],
                                 (n_tot_samples,),
                                 (args.sampling_period,),
                                 ids_type)

        xf = rfftfreq(dict_dataset['n_time_steps'], dict_dataset['dt_sec'])

        jj = 0
        start = 0
        # Reset lists:
        list_values = []
        list_target = []
        list_idx_time = []
        list_idx_inputs = []
        for i, (data, targets) in enumerate(tqdm(dict_dataset[subset + '_loader'])):
            assert data.shape[1] == dict_dataset['n_features']
            assert data.shape[2] == dict_dataset['n_inputs']

            if args.data_type == 'frequency':
                data = extract_interval(data, xf, args.bins_hist, args.center, args.span)
                assert data.shape[1] == args.bins_hist

            elif args.data_type == 'amplitude':
                data = extract_histogram(data, args.bins_hist, args.center, args.span)
                assert data.shape[1] == args.bins_hist

            elif args.data_type == 'slope':
                data = torch.diff(data, dim=1)
                data = extract_histogram(data, args.bins_hist, args.center, args.span)
                assert data.shape[1] == args.bins_hist

            elif args.data_type == 'current':
                # no preprocessing
                pass

            if i < (len(dict_dataset[subset + '_loader']) - 1): # last batch can be smaller than the batch size
                assert data.shape[0] == dict_dataset['batch_size']

            for i_batch in range(data.shape[0]):
                if args.data_type == 'current':
                    list_values.append(np.array(data[i_batch].to_sparse().values()))
                    list_idx_time.append(np.array(data[i_batch].to_sparse().indices()[0]))
                    list_idx_inputs.append(np.array(data[i_batch].to_sparse().indices()[1]))
                    assert (np.max(list_idx_time[-1]) <= dict_dataset['n_time_steps'])
                    assert (np.max(list_idx_inputs[-1]) <= dict_dataset['n_inputs'])
                else:
                    # No need to store values and indices separately since the samples are already 1-dim
                    list_values.append(np.array(data[i_batch]))
                    list_idx_time.append(np.array([]))
                    list_idx_inputs.append(np.array([]))
                    assert len(data[i_batch])>0

                list_target.append(np.array(targets[i_batch].item()))

                if (jj % args.sampling_period) == (args.sampling_period - 1):
                    with h5py.File(output_filename, 'a') as out:
                        #start = (jj // args.sampling_period) * args.sampling_period
                        out['values'][start:start + len(list_values)] = np.array(list_values, dtype=val_type)
                        out['idx_time'][start:start + len(list_values)] = np.array(list_idx_time, dtype=ids_type)
                        out['idx_inputs'][start:start + len(list_values)] = np.array(list_idx_inputs, dtype=ids_type)
                        out['targets'][start:start + len(list_values)] = np.array(list_target, dtype=ids_type)
                    start += len(list_values)
                    # Reset lists:
                    list_values = []
                    list_target = []
                    list_idx_time = []
                    list_idx_inputs = []

                # Increase counter samples
                jj += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train')

    parser.add_argument('--seed',
                        type=int,
                        default=10)
    parser.add_argument('--dataset',
                        type=str,
                        default='MNIST_compressed',
                        choices=['MNIST', 'Braille', 'MNIST_compressed'])
    parser.add_argument('--n_samples_train',
                        type=int,
                        default=6480)
    parser.add_argument('--n_samples_test',
                        type=int,
                        default=1620)
    parser.add_argument('--stim_len_sec',
                        type=float,
                        help='Duration (in sec) of input stimulus',
                        default=3)
    parser.add_argument('--dt_sec',
                        type=float,
                        help='Sampling frequency input current',
                        default=1e-2)
    parser.add_argument('--batch_size',
                        type=int,
                        default=128)
    parser.add_argument('--v_max',
                        type=float,
                        help='Parameter used for noise generation',
                        default=0.2)
    parser.add_argument('--center',
                        type=float,
                        default=0.1)
    parser.add_argument('--span',
                        type=float,
                        default=0.1)
    parser.add_argument('--sampling_period',
                        type=int,
                        help='Lenght chunks stored to disk',
                        default=1)
    parser.add_argument('--data_type',
                        type=str,
                        default='current',
                        choices=['current', 'frequency', 'amplitude', 'slope'])
    parser.add_argument('--bins_hist',
                        type=int,
                        default=100,
                        help='Bins histogram for datasets of amplitudes and slope')
    parser.add_argument('--home_dataset',
                        type=str,
                        help='Absolute path to output folder where the output dataset is stored',
                        default='home/mast/Progetti/tactile_encoding/data')  # './dataset/')#'/media/p308783/bics/Nicoletta/tactile_encoding/')
    args = parser.parse_args()

    main(args)
