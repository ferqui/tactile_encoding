"""
Generate .h5 file with samples from MNIST dataset saved as chunks with size one. This allows to deal with unlimited size
datasets as the output dataset is saved and updated at every chunk (of size 1), thus without requiring to accumulate
a list of samples before saving the full dataset at once.

Transform input MNIST samples to current across time (with noise).

- N.Risi
"""

import torch
import argparse
from utils_dataset import load_dataset, set_random_seed, extract_interval
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
import os
from numpy.fft import rfft, rfftfreq


def create_empty_dataset(h5py_file, list_dataset_names, shape, chunks, dtype):
    """
    Initialize input file with empty dataset.
    """
    for field in list_dataset_names:
        h5py_file.create_dataset(field, shape=shape, chunks=chunks, dtype=dtype)


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    generator = set_random_seed(args.seed, add_generator=True, device=device)
    shuffle_data = False

    # Load dataset:
    dict_dataset = load_dataset('MNIST',
                                batch_size=args.batch_size,
                                stim_len_sec=args.stim_len_sec,
                                dt_sec=args.dt_sec,
                                v_max=args.v_max,
                                generator=generator,
                                add_noise=True,
                                return_fft=args.data_type == 'frequency',
                                n_samples_train=args.n_samples_train,
                                n_samples_test=args.n_samples_test)

    path_to_dataset = Path(args.home_dataset)
    path_to_dataset.mkdir(parents=True, exist_ok=True)

    if args.data_type == 'current':
        filename_dataset = path_to_dataset.joinpath('MNIST',
                                                    args.data_type,
                                                    str(len(dict_dataset['train_loader'])) + '_' +
                                                    str(len(dict_dataset['test_loader'])))
    else:
        # Name dataset including info about center and span:
        filename_dataset = path_to_dataset.joinpath('MNIST',
                                                    args.data_type,
                                                    str(len(dict_dataset['train_loader'])) + '_' +
                                                    str(len(dict_dataset['test_loader'])) + '_c' +
                                                    str(args.center) + '_s' +
                                                    str(args.span))

    output_path = Path(filename_dataset)
    output_path.mkdir(parents=True, exist_ok=True)
    val_type = h5py.vlen_dtype(np.dtype('float32'))
    ids_type = h5py.vlen_dtype(np.dtype('int32'))

    for subset in ['train', 'test']:
        n_tot_samples = len(dict_dataset[subset + '_loader'])

        output_filename = str(filename_dataset.joinpath(subset + '.h5'))
        with h5py.File(output_filename, 'w') as out:
            out.attrs['batch_size'] = dict_dataset['batch_size']
            out.attrs['n_time_steps'] = dict_dataset['n_time_steps']
            out.attrs['n_freq_steps'] = dict_dataset['n_freq_steps']
            out.attrs['n_features'] = dict_dataset['n_features']
            out.attrs['n_classes'] = dict_dataset['n_classes']
            out.attrs['n_inputs'] = dict_dataset['n_inputs']
            out.attrs['dt_sec'] = dict_dataset['dt_sec']
            out.attrs['v_max'] = dict_dataset['v_max']
            out.attrs['sample_size'] = args.sample_size
            out.attrs['center'] = args.center
            out.attrs['span'] = args.span
            create_empty_dataset(out, ['values', 'idx_time', 'idx_inputs', 'targets'],
                                 (n_tot_samples,),
                                 (1,),
                                 ids_type)

        xf = rfftfreq(dict_dataset['n_time_steps'], dict_dataset['dt_sec'])

        for i, (data, targets) in enumerate(tqdm(dict_dataset[subset + '_loader'])):
            assert data.shape[0] == dict_dataset['batch_size']
            assert data.shape[1] == dict_dataset['n_features']
            assert data.shape[2] == dict_dataset['n_inputs']

            if args.data_type == 'frequency':
                # Extract interval with center and span:
                # (averages across n_inputs, reducing one dimension of data) and
                # downscales the feature dimension to samples_size
                data = extract_interval(data, xf, args.sample_size, args.center, args.span)
            elif args.data_type == 'amplitude':
                # TODO: Add preprocessing
                pass
            elif args.data_type == 'slope':
                # TODO: Add preprocessing
                pass
            elif args.data_type == 'current':
                # no preprocessing
                pass

            assert data.shape[0] == dict_dataset['batch_size']
            if args.data_type == 'current':
                pass  # data remains the same
            else:
                assert args.sample_size == data.shape[1]

            list_values = []
            list_target = []
            list_idx_time = []
            list_idx_inputs = []
            for i_batch in range(dict_dataset['batch_size']):
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
                list_target.append(np.array(targets[i_batch].item()))

                with h5py.File(output_filename, 'a') as out:
                    out['values'][i:i + 1] = np.array(list_values, dtype=val_type)
                    out['idx_time'][i:i + 1] = np.array(list_idx_time, dtype=ids_type)
                    out['idx_inputs'][i:i + 1] = np.array(list_idx_inputs, dtype=ids_type)
                    out['targets'][i:i + 1] = np.array(list_target, dtype=ids_type)
                list_values = []
                list_target = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train')

    parser.add_argument('--seed',
                        type=int,
                        default=10)
    parser.add_argument('--n_samples_train',
                        type=int,
                        default=-1)
    parser.add_argument('--n_samples_test',
                        type=int,
                        default=-1)
    parser.add_argument('--stim_len_sec',
                        type=float,
                        help='Duration (in sec) of input stimulus',
                        default=1)
    parser.add_argument('--dt_sec',
                        type=float,
                        help='Sampling frequency input current',
                        default=1e-2)
    parser.add_argument('--batch_size',
                        type=int,
                        default=1)
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
    parser.add_argument('--sample_size',
                        type=int,
                        default=10)
    parser.add_argument('--data_type',
                        type=str,
                        default='frequency',
                        choices=['current', 'frequency', 'amplitude', 'slope'])
    parser.add_argument('--home_dataset',
                        type=str,
                        help='Absolute path to output folder where the dataset is stored',
                        default='./dataset/')
    args = parser.parse_args()

    main(args)
