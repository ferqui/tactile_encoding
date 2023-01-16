import pandas as pd
import torch
import matplotlib.pyplot as plt
from time import localtime, strftime
import matplotlib
import argparse
import seaborn as sns

matplotlib.pyplot.ioff()  # turn off interactive mode
import numpy as np
import os
from training import MN_neuron
from datasets import load_analog_data
import time
import pickle
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

Current_PATH = os.getcwd()
# Set seed:
torch.manual_seed(0)

MNclass_to_param = {
    'A': {'a': 0, 'A1': 0, 'A2': 0},
    'C': {'a': 5, 'A1': 0, 'A2': 0}
}
class_labels = dict(zip(list(np.arange(len(MNclass_to_param.keys()))), MNclass_to_param.keys()))
inv_class_labels = {v: k for k, v in class_labels.items()}

def my_GWN(pars, mu, sig, myseed=False):
  """
  Function that generates Gaussian white noise input

  Args:
    pars       : parameter dictionary
    mu         : noise baseline (mean)
    sig        : noise amplitute (standard deviation)
    myseed     : random seed. int or boolean
                 the same seed will give the same
                 random number sequence

  Returns:
    I          : Gaussian white noise input
  """

  # Retrieve simulation parameters
  dt, range_t = pars['dt'], pars['range_t']
  Lt = range_t.size

  # Set random seed
  if myseed:
      np.random.seed(seed=myseed)
  else:
      np.random.seed()

  # Generate GWN
  # we divide here by 1000 to convert units to sec.
  I_gwn = mu + sig * np.random.randn(Lt) / np.sqrt(dt / 1000.)

  return I_gwn

def get_pca(X, Y, class_labels=None, exp_variance=None, fig_folder=None):
    # Standardize dataset:
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # Keep as many components needed to explain 95% of the variance:
    pca = PCA()
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    cum_variance = np.cumsum(pca.explained_variance_ratio_)


    # Plot cumulative variance
    fig = plt.figure()
    plt.plot(cum_variance, '.-')
    plt.ylim(0.0, 1.1)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative variance (%)')
    #fig.savefig(fig_folder.joinpath('cum_variance.pdf'), format='pdf')

    # Plot variance explained
    plt.figure()
    plt.plot(pca.explained_variance_ratio_, '.-')
    plt.xlabel('PCs')
    plt.ylabel('Variance ratio')

    # Relationship between PCs and input features:
    plt.figure(dpi=300)
    plt.matshow(pca.components_, cmap='viridis')  # ndarray of shape (n_components, n_features=n_time_bins)
    plt.ylabel('N components')
    plt.xlabel('N features')
    # plt.colorbar()
    plt.tight_layout()
    plt.show(block=False)

    # Plot dataset in low dimensional space:
    class_types = list(class_labels.values())
    cdict = {class_types[0]: 'red',
             class_types[1]: 'green'}
    marker = {class_types[0]: '*',
              class_types[1]: 'o'}
    alpha = {class_types[0]: .3,
             class_types[1]: .5}

    # 2D:
    Xax = X_pca[:, 0]
    Yax = X_pca[:, 1]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor('white')
    for label in class_types:
        ix = np.where(Y == label)
        ax.scatter(Xax[ix], Yax[ix], c=cdict[label], s=30,
                   label=label, marker=marker[label], alpha=alpha[label])
    ax.set_xlabel("PC1", fontsize=14)
    ax.set_ylabel("PC2", fontsize=14)
    ax.legend()

    # 3D:
    if X_pca.shape[1] >= 3:

        Zax = X_pca[:, 2]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('white')
        for label in class_types:
            ix = np.where(Y == label)
            ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[label], s=30,
                       label=label, marker=marker[label], alpha=alpha[label])
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel("PC2", fontsize=14)
        ax.set_zlabel("PC3", fontsize=14)
        ax.view_init(100, -50)
        ax.legend()
        plt.show()
        #fig.savefig(fig_folder.joinpath('PCA_3D.pdf'), format='pdf')

    return X_pca, cum_variance  # Return principal components

def get_input_step_current(dt_sec=0.001, stim_length_sec=0.1, amplitudes=np.arange(10),
                           sig=.1, n_trials=10):
    """
    Return amplitude of input current across time, with as many input signals as the dimension of
    the input amplitudes.

    dt_sec:
    stim_length_sec:
    amplitudes
    """
    n_time_bins = int(np.floor(stim_length_sec / dt_sec))
    n_neurons = len(amplitudes)*n_trials
    stim = []
    for a in amplitudes:
        for n in range(n_trials):
            #stim.append(torch.tensor([a] * n_time_bins))
            I_gwn = a + sig * np.random.randn(n_time_bins) / np.sqrt(n_time_bins / 1000.)
            stim.append(torch.tensor(I_gwn))

    input_current = torch.stack(stim, dim=1)
    input_current = torch.reshape(input_current, (n_time_bins, n_neurons))
    input_current = input_current[None, :]  # add first dimension for batch size 1

    assert input_current.shape[0] == 1
    assert input_current.shape[1] == n_time_bins
    assert input_current.shape[2] == len(amplitudes)*n_trials

    return input_current

def pca_isi(dict_spk_rec, class_labels):

    class_types = dict_spk_rec.keys()
    feature_to_col_id = ['n_spikes', 'std_isi', 'entropy_isi']
    n_features = len(feature_to_col_id)  # n_statistics_isi

    list_statistics_isi = []
    list_labels = []
    for cc in class_types:

        statistics_isi = torch.zeros((dict_spk_rec[list(class_types)[0]][0].shape[1], n_features))  # n_neurons (=trials per model=current ampl) x n_features

        # Reduce batch (first) dimension:
        spike_events = dict_spk_rec[cc][0]
        t_spike_in_dt = np.arange(spike_events.shape[0])

        # Extract ISI statistics:

        for idx_trial in range(spike_events.shape[1]):

            spike_times = t_spike_in_dt[np.where(spike_events[:,idx_trial].int())[0]]
            if len(spike_times)>=2:
                isi = np.diff(spike_times)
                H_isi = entropy(isi, base=2)
                std_isi = np.std(isi)
                statistics_isi[idx_trial, feature_to_col_id.index('n_spikes')] = len(spike_times)
                statistics_isi[idx_trial, feature_to_col_id.index('std_isi')] = std_isi
                statistics_isi[idx_trial, feature_to_col_id.index('entropy_isi')] = H_isi
            else:
                statistics_isi[idx_trial, feature_to_col_id.index('n_spikes')] = 0
                statistics_isi[idx_trial, feature_to_col_id.index('std_isi')] = 0
                statistics_isi[idx_trial, feature_to_col_id.index('entropy_isi')] = 0

            list_labels.append(cc)
        list_statistics_isi.append(statistics_isi)

    X = torch.cat(list_statistics_isi, dim=0)
    Y = np.array(list_labels)

    X_pca, cum_variance = get_pca(X, Y, class_labels)

    return X_pca

def plot_outputs(dict_spk_rec, mem_rec, dt):
    dict_mem_rec = dict.fromkeys(mem_rec.keys(), [])
    dict_isi = dict.fromkeys(mem_rec.keys(), [])

    for key in mem_rec.keys():
        n_current_values = mem_rec[key][0].shape[1]
        dict_mem_rec[key] = {'time': [], 'vmem': [], 'Ie': []}
        tmp = mem_rec[key][0]
        dict_isi[key] = {}
        for a in range(tmp.shape[1]):
            dict_isi[key][a] = {'time': [], 'isi': []}
            vmem = np.array(tmp[:, a])
            dict_mem_rec[key]['time'].extend(list(np.arange(len(vmem))))
            dict_mem_rec[key]['vmem'].extend(list(vmem))
            dict_mem_rec[key]['Ie'].extend([a] * len(vmem))

            t_spike_in_dt = np.arange(len(vmem))
            idx_spike = np.where(np.array(dict_spk_rec[key][0][:, a]).astype(int))[0]
            if len(t_spike_in_dt):
                t_spike_in_dt = t_spike_in_dt[idx_spike]
                dict_isi[key][a]['time'] = t_spike_in_dt[:-1]
                dict_isi[key][a]['isi'] = np.diff(t_spike_in_dt)
            else:
                dict_isi[key][a]['time'] = []
                dict_isi[key][a]['isi'] = []

    palette = sns.cubehelix_palette(n_colors=n_current_values)
    fig, axs = plt.subplots(3, len(dict_spk_rec.keys()), sharex=True)
    for i, neuron_type in enumerate(dict_spk_rec.keys()):
        axs[0, i].imshow(np.transpose(dict_spk_rec[neuron_type][0]), cmap='Greys',
                         interpolation='none', aspect='auto')
        sns.lineplot(data=pd.DataFrame(dict_mem_rec[neuron_type]), x='time', y='vmem', hue='Ie',
                     ax=axs[2, i], palette=palette)
        axs[2, i].set_xlabel('Time (ms)')
        axs[0, i].set_title('MN type: ' + neuron_type)

        for a in range(mem_rec[neuron_type][0].shape[1]):
            axs[1, i].plot(dict_isi[neuron_type][a]['time'], dict_isi[neuron_type][a]['isi'], 'o', markersize=2,
                           color=palette[a])

    axs[1, 0].set_ylabel('ISI (ms)')
    axs[2, 0].set_ylabel('Vmem')
    axs[0, 0].set_ylabel('Ie')
    axs[2, 1].set_ylabel('')
    fig.set_size_inches(10, 9)
    fig.savefig('Output_MN.pdf', format='pdf')
    plt.show()

def main(args):
    exp_id = strftime("%d%b%Y_%H-%M-%S", localtime())

    list_classes = args.MNclasses_to_test
    nb_inputs = args.nb_inputs
    amplitudes = np.arange(1, nb_inputs + 1)  # TODO: Find correct offset
    n_repetitions = args.n_repetitions
    sigma = args.sigma

    # each neuron receives a different input amplitude
    dict_spk_rec = dict.fromkeys(list_classes, [])
    dict_mem_rec = dict.fromkeys(list_classes, [])
    for MN_class_type in list_classes:
        neurons = MN_neuron(len(amplitudes)*n_repetitions, MNclass_to_param[MN_class_type], dt=args.dt, train=False)

        x_local = get_input_step_current(dt_sec=args.dt, stim_length_sec=args.stim_length_sec, amplitudes=amplitudes,
                                         n_trials=n_repetitions, sig=sigma)

        neurons.reset()
        spk_rec = []
        mem_rec = []
        for t in range(x_local.shape[1]):
            out = neurons(x_local[:, t])

            spk_rec.append(neurons.state.spk)
            mem_rec.append(neurons.state.V)

        dict_spk_rec[MN_class_type] = torch.stack(spk_rec, dim=1) # shape: batch_size, time_steps, neurons (i.e., current amplitudes)
        dict_mem_rec[MN_class_type] = torch.stack(mem_rec, dim=1)

    plot_outputs(dict_spk_rec, dict_mem_rec, dt=args.dt)

    X_pca_isi = pca_isi(dict_spk_rec, class_labels)

    print('End')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('TODO')
    parser.add_argument('--MNclasses_to_test', type=list, default=['A', 'C'], help="learning rate")
    parser.add_argument('--nb_inputs', type=int, default=10)
    parser.add_argument('--n_repetitions', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=0, help='sigma gaussian distrubution of I current')
    parser.add_argument('--stim_length_sec', type=float, default=0.2)
    parser.add_argument('--selected_input_channel', type=int, default=0)
    parser.add_argument('--dt', type=float, default=0.001)

    args = parser.parse_args()

    main(args)
