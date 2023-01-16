import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd


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
    fig.savefig(fig_folder.joinpath('cum_variance.pdf'), format='pdf')

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
    fig.savefig(fig_folder.joinpath('PCA_2D.pdf'), format='pdf')

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
        fig.savefig(fig_folder.joinpath('PCA_3D.pdf'), format='pdf')

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
    n_neurons = len(amplitudes) * n_trials
    stim = []
    list_mean_current = [] #list with mean current value (same dimension as the last dimension of input_current)
    for a in amplitudes:
        for n in range(n_trials):
            # stim.append(torch.tensor([a] * n_time_bins))
            I_gwn = a + sig * np.random.randn(n_time_bins) / np.sqrt(n_time_bins / 1000.)
            stim.append(torch.tensor(I_gwn))
            list_mean_current.append(a)

    input_current = torch.stack(stim, dim=1)
    input_current = torch.reshape(input_current, (n_time_bins, n_neurons))
    input_current = input_current[None, :]  # add first dimension for batch size 1

    assert input_current.shape[0] == 1
    assert input_current.shape[1] == n_time_bins
    assert input_current.shape[2] == len(amplitudes) * n_trials # thid dim: n_trials = n_neurons (all stimulated ad once)

    return input_current, list_mean_current


def pca_isi(dict_spk_rec, class_labels, fig_folder=None):
    class_types = dict_spk_rec.keys()
    feature_to_col_id = ['n_spikes', 'std_isi', 'entropy_isi']
    n_features = len(feature_to_col_id)  # n_statistics_isi

    list_statistics_isi = []
    list_labels = []
    for cc in class_types:

        statistics_isi = torch.zeros((dict_spk_rec[list(class_types)[0]][0].shape[1],
                                      n_features))  # n_neurons (=trials per model=current ampl) x n_features

        # Reduce batch (first) dimension:
        spike_events = dict_spk_rec[cc][0]
        t_spike_in_dt = np.arange(spike_events.shape[0])

        # Extract ISI statistics:
        for idx_trial in range(spike_events.shape[1]):

            spike_times = t_spike_in_dt[np.where(spike_events[:, idx_trial].int())[0]]
            if len(spike_times) >= 2:
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

    X_pca, cum_variance = get_pca(X, Y, class_labels, fig_folder=fig_folder)

    return X_pca


def plot_outputs(dict_spk_rec, mem_rec, list_mean_current, xlim=None, fig_folder=None):
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
            dict_mem_rec[key]['vmem'].extend(list(vmem*1e3)) # from V to mV
            dict_mem_rec[key]['Ie'].extend([list_mean_current[a]] * len(vmem))

            t_spike_in_dt = np.arange(len(vmem))
            idx_spike = np.where(np.array(dict_spk_rec[key][0][:, a]).astype(int))[0]
            if len(t_spike_in_dt):
                t_spike_in_dt = t_spike_in_dt[idx_spike]
                dict_isi[key][a]['time'] = t_spike_in_dt[:-1]
                dict_isi[key][a]['isi'] = np.diff(t_spike_in_dt)
            else:
                dict_isi[key][a]['time'] = []
                dict_isi[key][a]['isi'] = []

    palette = sns.cubehelix_palette(n_colors=len(np.unique(list_mean_current)))
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
                           color=palette[list(np.unique(list_mean_current)).index(list_mean_current[a])])

    axs[1, 0].set_ylabel('ISI (ms)')
    axs[2, 0].set_ylabel('Vmem (mV)')
    axs[0, 0].set_ylabel('Ie (V/s)')
    axs[2, 1].set_ylabel('')
    if xlim:
        axs[0,0].set_xlim(xlim)

    fig.align_ylabels(axs[:])
    fig.set_size_inches(20, 9)
    fig.savefig(fig_folder.joinpath('Output_MN.pdf'), format='pdf')
    plt.show()


def plot_vmem(dict_spk_rec, mem_rec, list_mean_current, xlim=None, fig_folder=None):
    dict_mem_rec = dict.fromkeys(mem_rec.keys(), [])

    for key in mem_rec.keys():
        n_current_values = mem_rec[key][0].shape[1]
        dict_mem_rec[key] = {'time': [], 'vmem': [], 'Ie': []}
        tmp = mem_rec[key][0]
        for a in range(tmp.shape[1]):
            vmem = np.array(tmp[:, a])
            dict_mem_rec[key]['time'].extend(list(np.arange(len(vmem))))
            dict_mem_rec[key]['vmem'].extend(list(vmem * 1e3))  # from V to mV
            dict_mem_rec[key]['Ie'].extend([list_mean_current[a]] * len(vmem))

    palette = sns.cubehelix_palette(n_colors=len(np.unique(list_mean_current)))
    fig, axs = plt.subplots(1, len(dict_spk_rec.keys()), sharex=True)
    for i, neuron_type in enumerate(dict_spk_rec.keys()):
        sns.lineplot(data=pd.DataFrame(dict_mem_rec[neuron_type]), x='time', y='vmem', hue='Ie',
                     ax=axs[i], palette=palette)
        axs[i].set_xlabel('Time (ms)')
        axs[i].set_title('MN type: ' + neuron_type)

    axs[0].set_ylabel('Vmem (mV)')
    if xlim:
        axs[0].set_xlim(xlim)

    fig.align_ylabels(axs[:])
    fig.set_size_inches(15, 6)
    fig.savefig(fig_folder.joinpath('Vmem.pdf'), format='pdf')
    plt.show()
