import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import argparse

"""
Plot input samples generated with test_dataloader_h5file.py with type_data = 'current'
"""


def main(args):
    folder = Path('dataset_analysis')
    folder_dataset = Path('dataset')
    folder_run = Path(os.path.join(folder, 'MNIST'))
    folder_fig = folder_run.joinpath('fig')

    n_samples = 10
    folder_name = f'{args.n_samples_train}_{args.n_samples_test}'
    template = [str(i) for i in range(n_samples)]
    path_to_folder = './' + str(folder_dataset.joinpath('MNIST', 'current',
                                                                                f'{args.n_samples_train}_{args.n_samples_test}'))
    list_data = [file for file in os.listdir(path_to_folder)
                 if file[1] in template]

    fig, axs = plt.subplots(2, n_samples)
    for i, file in enumerate(list_data):
        data = np.load(path_to_folder + '/' + file)

        axs[1, i].imshow(np.transpose(data), interpolation='none', aspect='auto')
        axs[0, i].imshow(np.sum(data, axis=0).reshape(28, 28), interpolation='none', aspect='equal', cmap='Greys')
        axs[0, i].set_title(f'{file[-5]}')
        if not (i == 0):
            axs[0, i].set_yticks([])
            axs[1, i].set_yticks([])
    fig.set_size_inches(20, 5)
    fig.savefig(folder_fig.joinpath(f'./plot_samples_{folder_name}.pdf'))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('plot_dataloader')
    parser.add_argument('--n_samples_train',
                        type=int,
                        default=500)
    parser.add_argument('--n_samples_test',
                        type=int,
                        default=10)

    args = parser.parse_args()

    # Run:
    main(args)
