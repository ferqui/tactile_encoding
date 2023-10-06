import os
import torch
import numpy as np
import argparse
import tqdm


def main(args):
    if args.data_type == 'frequency':
        centers = np.linspace(0.5, 5, args.n_samples_sweep)
        spans = np.linspace(0.5, 1, args.n_samples_sweep)
        sweep = tqdm.tqdm(total=len(centers) * len(spans), desc=f"Frequency Sweeping", position=0, leave=True)

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
        # TODO:
        print('Not implemented yet')
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('dataloader')

    parser.add_argument('--data_type',
                        type=str,
                        default='frequency',
                        choices=['current', 'frequency', 'amplitude', 'slope'])
    parser.add_argument('--n_samples_train',
                        type=int,
                        default=-1)
    parser.add_argument('--sample_size',
                        type=int,
                        default=10)
    parser.add_argument('--home_dataset',
                        type=str,
                        help='Absolute path to output folder where the dataset is stored',
                        default='./dataset/')
    parser.add_argument('--n_samples_sweep',
                        type=int,
                        default=5)
    args = parser.parse_args()

    # Run:
    main(args)
