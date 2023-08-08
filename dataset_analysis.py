import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
import seaborn as sns
import numpy as np

from datasets import load_data
from parameters.MN_params import MNparams_dict, INIT_MODE
from models import Encoder, LIF_neuron, MN_neuron
from auxiliary import compute_classification_accuracy, plot_spikes, plot_voltages

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from numpy.fft import rfft, rfftfreq

firing_mode_dict = {
    "FA": {"a": 5, "A1": 0, "A2": 0},
    "SA": {"a": 0, "A1": 0, "A2": 0},
    "MIX": {"a": 5, "A1": 5, "A2": -0.3},
}


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    folder_run = Path('datset_analysis')
    folder_fig = folder_run.joinpath('fig')
    folder_fig.mkdir(parents=True, exist_ok=True)

    ###########################################
    ##                Dataset                ##
    ###########################################
    upsample_fac = 1
    dt = (1 / 100.0) / upsample_fac
    file_name = "data/data_braille_letters_all.pkl"
    data, labels, _, _, _, _ = load_data(file_name, upsample_fac)
    vmax = torch.max(data).item()
    vmin = torch.min(data).item()
    nb_channels = data.shape[-1]

    # Find the first occurrence of the unique values in the list of classes (one sample for each letter)
    idx_letters, indices = np.unique(labels, return_index=True)

    # Amplitude:
    n_rows = 4
    n_cols = 7
    fig, axs = plt.subplots(n_rows, n_cols, sharey=True, sharex=True)
    for i, idx_to_plot in zip(idx_letters, indices):
        ax = axs[i // n_cols, i % n_cols]
        im = ax.imshow(np.transpose(data[idx_to_plot, :, :]), aspect='auto', interpolation='none', vmax=vmax, vmin=vmin)
        ax.set_title(i)

    for i in range(n_rows):
        axs[i, 0].set_ylabel('Channels')

    for i in range(n_cols):
        axs[n_rows - 1, i].set_xlabel('Time bins')

    cbaxes = fig.add_axes([0.85, 0.3, 0.03, 0.5])
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5, cax = cbaxes, label='Amplitude')

    axs[n_rows - 1, n_cols - 1].set_axis_off()
    fig.set_size_inches(10, 8)
    fig.subplots_adjust(wspace=0.3, hspace=0.3, right=0.8)
    fig.savefig(folder_fig.joinpath('Amplitudes.pdf'), format='pdf', dpi=300)

    # ------------------------------------------------------------------------------------------ Frequency:
    fig, axs = plt.subplots(n_rows, n_cols, sharey=True, sharex=True)
    n_samples = data.shape[1]  # number of time steps
    for i, idx_to_plot in zip(idx_letters, indices):
        ax = axs[i // n_cols, i % n_cols]
        x = data[idx_to_plot, :, :]
        xf = rfftfreq(n_samples, dt)
        yf = rfft(x, axis=0)

        n_channels = yf.shape[1]
        n_freq = yf.shape[0]
        freq = np.array([[i]*n_channels for i in xf]).flatten()
        yf = np.abs(yf).flatten()
        ch = np.repeat(np.array([np.arange(n_channels)]), repeats=n_freq, axis=0).flatten()
        df = pd.DataFrame({'freq': freq, 'y': yf, 'ch': ch})
        sns.lineplot(data=df, x='freq', y='y',hue='ch', ax=ax)
        ax.set_title(i)

        if i == (len(idx_letters)-1):
            sns.move_legend(ax, "right", bbox_to_anchor=(2.5, 0.5))
        else:
            ax.get_legend().remove()

    for i in range(n_rows):
        axs[i, 0].set_ylabel('|yf|')

    for i in range(n_cols):
        axs[n_rows - 1, i].set_xlabel('Freq (Hz)')

    axs[n_rows - 1, n_cols - 1].set_axis_off()
    fig.set_size_inches(10, 8)
    fig.subplots_adjust(wspace=0.3, hspace=0.3, right=0.8)
    fig.savefig(folder_fig.joinpath('Frequency.pdf'), format='pdf', dpi=300)

    # ------------------------------------------------------------------------------------------ Frequency - zoom in :
    fig, axs = plt.subplots(n_rows, n_cols, sharey=True, sharex=True)
    n_samples = data.shape[1]  # number of time steps
    for i, idx_to_plot in zip(idx_letters, indices):
        ax = axs[i // n_cols, i % n_cols]

        x = data[idx_to_plot, :, :]
        xf = rfftfreq(n_samples, dt)
        yf = rfft(x, axis=0)
        n_channels = yf.shape[1]
        n_freq = yf.shape[0]
        freq = np.array([[i]*n_channels for i in xf]).flatten()
        yf = np.abs(yf).flatten()
        ch = np.repeat(np.array([np.arange(n_channels)]), repeats=n_freq, axis=0).flatten()

        df = pd.DataFrame({'freq': freq, 'y': yf, 'ch': ch})
        sns.lineplot(data=df, x='freq', y='y',hue='ch', ax=ax)
        ax.set_title(i)

        if i == (len(idx_letters)-1):
            sns.move_legend(ax, "right", bbox_to_anchor=(2.5, 0.5))
        else:
            ax.get_legend().remove()

    for i in range(n_rows):
        axs[i, 0].set_ylabel('|yf|')

    for i in range(n_cols):
        axs[n_rows - 1, i].set_xlabel('Freq (Hz)')

    axs[n_rows - 1, n_cols - 1].set_axis_off()
    ax.set_xlim(0,10)
    fig.set_size_inches(10, 8)
    fig.subplots_adjust(wspace=0.3, hspace=0.3, right=0.8)
    fig.savefig(folder_fig.joinpath('Frequency_zoomed_in.pdf'), format='pdf', dpi=300)

    # ------------------------------------------------------------------------------------------ Slope :
    fig, axs = plt.subplots(n_rows, n_cols, sharey=True, sharex=True)
    n_samples = data.shape[1]  # number of time steps
    for i, idx_to_plot in zip(idx_letters, indices):
        ax = axs[i // n_cols, i % n_cols]

        x = data[idx_to_plot, :, :]
        dy_dx = np.diff(x, axis=0)/dt
        n_channels = x.shape[1]
        time_bins = np.array([[i]*n_channels for i in range(n_samples-1)]).flatten()
        dy_dx = dy_dx.flatten()
        ch = np.repeat(np.array([np.arange(n_channels)]), repeats=n_samples-1, axis=0).flatten()
        df = pd.DataFrame({'Time bins': time_bins, 'dy_dx': dy_dx, 'ch': ch})

        sns.lineplot(data=df, x='Time bins', y='dy_dx', hue='ch', ax=ax)
        ax.set_title(i)

        if i == (len(idx_letters)-1):
            sns.move_legend(ax, "right", bbox_to_anchor=(2.5, 0.5))
        else:
            ax.get_legend().remove()

    for i in range(n_rows):
        axs[i, 0].set_ylabel('dy/dx')

    for i in range(n_cols):
        axs[n_rows - 1, i].set_xlabel('Time bins')

    axs[n_rows - 1, n_cols - 1].set_axis_off()
    fig.set_size_inches(10, 8)
    fig.subplots_adjust(wspace=0.3, hspace=0.3, right=0.8)
    fig.savefig(folder_fig.joinpath('Slopes.pdf'), format='pdf', dpi=300)

    # ------------------------------------------------------------------------------------------ Slope distribution:
    fig, axs = plt.subplots(n_rows, n_cols, sharey=True, sharex=True)
    n_samples = data.shape[1]  # number of time steps
    for i, idx_to_plot in zip(idx_letters, indices):
        ax = axs[i // n_cols, i % n_cols]

        x = data[idx_to_plot, :, :]
        dy_dx = np.diff(x, axis=0)/dt
        n_channels = x.shape[1]
        time_bins = np.array([[i]*n_channels for i in range(n_samples-1)]).flatten()
        dy_dx = dy_dx.flatten()
        ch = np.repeat(np.array([np.arange(n_channels)]), repeats=n_samples-1, axis=0).flatten()
        df = pd.DataFrame({'Time bins': time_bins, 'dy_dx': dy_dx, 'ch': ch})

        sns.histplot(data=df, x='dy_dx', ax=ax)
        ax.set_title(i)

    for i in range(n_rows):
        axs[i, 0].set_ylabel('Count')

    for i in range(n_cols):
        axs[n_rows - 1, i].set_xlabel('Slopes')

    axs[n_rows - 1, n_cols - 1].set_axis_off()
    fig.set_size_inches(10, 8)
    fig.subplots_adjust(wspace=0.3, hspace=0.3, right=0.8)
    fig.savefig(folder_fig.joinpath('Slopes_dist.pdf'), format='pdf', dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('dataset_analysis')
    parser.add_argument('--stim_length_sec', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--dt', type=float, default=0.001)

    args = parser.parse_args()

    main(args)
