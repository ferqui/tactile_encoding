import os
import torch
import numpy as np
import argparse
import tqdm


def main(args):

    if args.data_type == 'frequency':
        centers = np.linspace(10, 20, args.n_samples_sweep)
        spans = np.linspace(10, 20, args.n_samples_sweep)
        sweep = tqdm.tqdm(total=len(centers) * len(spans), desc=f"{args.data_type} Sweeping", position=0, leave=True)

        for center in centers:
            for span in spans:
                cmd = (f'python3 generate_MNIST_dataset.py --data_type {args.data_type} '
                       f'--center {center} --span {span} --n_samples_train {args.n_samples_train} '
                       f'--sample_size {args.sample_size} --home_dataset {args.home_dataset}')
                os.system(cmd)
                sweep.update()

    elif args.data_type == 'amplitude':
        centers = np.linspace(0.5, 100, args.n_samples_sweep)
        spans = np.linspace(0.5, 10, args.n_samples_sweep)
        sweep = tqdm.tqdm(total=len(centers) * len(spans), desc=f"{args.data_type} Sweeping", position=0, leave=True)

        for center in centers:
            for span in spans:
                cmd = (f'python3 generate_MNIST_dataset.py --data_type {args.data_type} '
                       f'--center {center} --span {span} --n_samples_train {args.n_samples_train} '
                       f'--sample_size {args.sample_size} --home_dataset {args.home_dataset}')
                os.system(cmd)
                sweep.update()

    elif args.data_type == 'slope':
        centers = np.linspace(0.5, 5, args.n_samples_sweep)
        spans = np.linspace(0.5, 1, args.n_samples_sweep)
        sweep = tqdm.tqdm(total=len(centers) * len(spans), desc=f"{args.data_type} Sweeping", position=0, leave=True)

        for center in centers:
            for span in spans:
                cmd = (f'python3 generate_MNIST_dataset.py --data_type {args.data_type} '
                       f'--center {center} --span {span} --n_samples_train {args.n_samples_train} '
                       f'--sample_size {args.sample_size} --home_dataset {args.home_dataset}')
                os.system(cmd)
                sweep.update()

    elif args.data_type == 'current':
        cmd = (f'python3 generate_MNIST_dataset.py --data_type {args.data_type} '
               f'--n_samples_train {args.n_samples_train} --home_dataset {args.home_dataset}')
        os.system(cmd)

    else:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser('dataloader')

    parser.add_argument('--data_type',
                        type=str,
                        default='frequency',
                        choices=['current', 'frequency', 'amplitude', 'slope'])
    parser.add_argument('--idx_job_array',
                        type=int,
                        default=None)
    parser.add_argument('--n_samples_train',
                        type=int,
                        default=20000)
    parser.add_argument('--sample_size',
                        type=int,
                        default=10)
    parser.add_argument('--home_dataset',
                        type=str,
                        help='Absolute path to output folder where the dataset is stored',
                        default='/media/p308783/bics/Nicoletta/tactile_encoding/')
    parser.add_argument('--n_samples_sweep',
                        type=int,
                        default=5)
    args = parser.parse_args()

    if args.idx_job_array is not None:
        print('Lunched script with HPC - using job array ID to select data type')
        # overwrite input argument --data_type:
        array_data_types = ['current', 'frequency', 'amplitude', 'slope']
        d = vars(args)  # copy by reference (checked below)
        key = 'data_type'
        d[key] = array_data_types[args.idx_job_array-1]
        assert (args.__dict__[key] == d[key])

        print(f'Running for data type: {d[key]}')
        print(f' with args: ')
        for key, value in d.items():
            print(f'{key}: {value}')

    # Run:
    main(args)
