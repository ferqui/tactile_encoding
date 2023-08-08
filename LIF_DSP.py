import torch
from torch.utils.data import DataLoader
from torchnmf.nmf import NMF
import matplotlib.pyplot as plt
from time import localtime, strftime
from tqdm import trange
from utils import addToNetMetadata, addHeaderToMetadata, set_results_folder, generate_dict
import matplotlib
import torchviz

matplotlib.pyplot.ioff()  # turn off interactive mode
import numpy as np
import os
import models
from datasets import load_analog_data
import time
import pickle
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split

Current_PATH = os.getcwd()
# matplotlib.use('Agg')
# Set seed:
torch.manual_seed(0)

# ---------------------------- Input -----------------------------------
save_out = True  # Flag to save figures:
multiprocess = False
LIFclasses = {
    'adapt': {'beta_adapt': 20, 'b_0': 0.1},
    'noadapt': {'beta_adapt': 0, 'b_0': 0.1},
}

ranges = {
    # 'ampli':[np.linspace(1, 4, 10),np.linspace(1, 10, 10),np.linspace(1, 100, 10)],
    # 'freq_pos':[np.linspace(10,100,10),np.linspace(10,500,10),np.linspace(10,1000,10)],
    # 'freq_neg': [np.linspace(10, 100, 10), np.linspace(10, 500, 10), np.linspace(10, 1000, 10)],
    # 'ampli_neg':[np.linspace(1, 4, 10),np.linspace(1, 10, 10),np.linspace(1, 100, 10)],
    'slopes': [
        np.linspace(1, 4, 10),
        # np.linspace(1, 0.4, 10),
        # np.linspace(1, 0.04, 10)
    ]
}
encoding_methods = ['spike']  # ,'count', 'isi']
run_with_fake_input = False
# ---------------------------- Parameters -----------------------------------
threshold = "enc"
run = "_3"

file_dir_params = 'parameters/'
param_filename = 'parameters_th' + str(threshold)
file_name_parameters = file_dir_params + param_filename + '.txt'
params = {}
with open(file_name_parameters) as file:
    for line in file:
        (key, value) = line.split()
        if key == 'time_bin_size' or key == 'nb_input_copies' or key == 'n_param_values' or key == 'min_range' or key == 'max_range':
            params[key] = int(value)
        else:
            params[key] = np.double(value)

# variable_range = np.linspace(params['min_range'], params['max_range'], params['n_param_values'])


# ----------------------- Experiment Folders ----------------------------
# Experiment name:
exp_id = strftime("%d%b%Y_%H-%M-%S", localtime())

fig3, ax3 = plt.subplots(len(LIFclasses), 1)
fig4, ax4 = plt.subplots(len(LIFclasses), 1)
fig5, ax5 = plt.subplots()

h = 0


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def run(dataset, device, neuron, varying_element, rank_NMF, model, training={}, list_loss=[],
        results_dir=None, net=None, neuron_id=4, epoch=0):
    return list_loss, net


def sweep_steps(amplitudes=np.arange(10), n_trials=10, dt_sec=0.001, stim_length_sec=0.1, sig=.1, debug_plot=True):
    """
    Return amplitude of input current across time, with as many input signals as the dimension of
    the input amplitudes.

    dt_sec:
    stim_length_sec:
    amplitudes

    input_curent: batch_size (1) x time bins x neurons (or n amplitudes x n trials)
    """
    n_time_bins = int(np.floor(stim_length_sec / dt_sec))
    n_neurons = len(amplitudes) * n_trials
    stim = []
    list_mean_current = []  # list with mean current value (same dimension as the last dimension of input_current)
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
    assert input_current.shape[2] == len(
        amplitudes) * n_trials  # thid dim: n_trials = n_neurons (all stimulated ad once)

    if debug_plot:
        for i in range(input_current.shape[2]):
            plt.plot(np.arange(n_time_bins) * dt_sec, input_current[0, :, i].cpu())
        plt.xlabel('Time (sec)')
        plt.ylabel('Input current')

    return input_current, list_mean_current


def sweep_slopes(slopes=np.arange(10), n_trials=10, dt_sec=0.001, stim_length_sec=0.1, sig=.1, debug_plot=True, last=5,
                 first=0):
    """
    Return amplitude of input current across time, with as many input signals as the dimension of
    the input amplitudes.

    dt_sec:
    stim_length_sec:
    amplitudes

    input_curent: batch_size (1) x time bins x neurons (or n amplitudes x n trials)
    """
    n_time_bins = int(np.floor(stim_length_sec / dt_sec))
    n_neurons = len(slopes) * n_trials
    stim = []
    list_mean_current = []
    slopes = slopes / dt_sec
    # list with mean current value (same dimension as the last dimension of input_current)
    for a in slopes:
        time_necessary = (last - first) / a
        time_first = (stim_length_sec - time_necessary) / 2
        time_last = (stim_length_sec - time_necessary) / 2
        assert time_first > dt_sec
        assert time_necessary > dt_sec
        first_vec = np.linspace(first, first, int(np.floor(time_first / dt_sec)))
        last_vec = np.linspace(last, last, int(np.floor(time_last / dt_sec)))
        slope = np.linspace(0, last - first, int(np.floor(time_necessary / dt_sec)))
        # slope = np.arange(0,last-first,time_necessary/dt_sec)
        for n in range(n_trials):
            # stim.append(torch.tensor([a] * n_time_bins))
            try:

                value = np.concatenate([first_vec, slope, last_vec])
                remaining = np.linspace(last, last, n_time_bins - len(value))
                value = np.concatenate([value, remaining])
                I_gwn = value + sig * np.random.randn(n_time_bins) / np.sqrt(n_time_bins / 1000.)
            except ValueError:
                print('ciao')
            stim.append(torch.tensor(I_gwn))
            list_mean_current.append(a)

    input_current = torch.stack(stim, dim=1)
    input_current = torch.reshape(input_current, (n_time_bins, n_neurons))
    input_current = input_current[None, :]  # add first dimension for batch size 1

    assert input_current.shape[0] == 1
    assert input_current.shape[1] == n_time_bins
    assert input_current.shape[2] == len(
        slopes) * n_trials  # thid dim: n_trials = n_neurons (all stimulated ad once)

    if debug_plot:
        for i in range(input_current.shape[2]):
            plt.plot(np.arange(n_time_bins) * dt_sec, input_current[0, :, i].cpu())
        plt.xlabel('Time (sec)')
        plt.ylabel('Input current')
    unique_slopes = np.unique(list_mean_current)
    list_mean_slopes_index = []
    for mean_slopes in list_mean_current:
        list_mean_slopes_index.append(np.where(unique_slopes == mean_slopes)[0][0])
    return input_current, list_mean_slopes_index


def sweep_amplitude_oscillations(amplitudes=np.arange(10), n_trials=10, offset=0, f=10, fs=1000, target_snr_db=20,
                                 debug_plot=True, add_noise=True):
    """
    Return amplitude of input current across time, with as many input signals as the dimension of
    the input amplitudes.

    dt_sec:
    stim_length_sec:
    amplitudes
    """
    t = np.arange(fs) / fs
    print(f'n time bins:{len(t)}')
    print(f'Period T:{1 / f} sec')

    n_time_bins = len(t)
    n_neurons = len(amplitudes) * n_trials
    stim = []
    list_mean_current = []  # list with mean current value (same dimension as the last dimension of input_current)
    for a in amplitudes:
        for n in range(n_trials):
            x = a * np.sin(2 * np.pi * f * t) + offset
            x_watts = x ** 2

            sig_avg_watts = np.mean(x_watts)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            # Calculate noise according to [2] then convert to watts
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            # Generate an sample of white noise with mean zero. For white noise, Ex and the average power is then equal to the variance Ex.
            mean_noise = 0
            noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
            # Noise up the original signal
            I = x + noise_volts * add_noise

            stim.append(torch.tensor(I))
            list_mean_current.append(a)

    input_current = torch.stack(stim, dim=1)
    input_current = torch.reshape(input_current, (n_time_bins, n_neurons))
    input_current = input_current[None, :]  # add first dimension for batch size 1

    assert input_current.shape[0] == 1
    assert input_current.shape[1] == n_time_bins
    assert input_current.shape[2] == len(
        amplitudes) * n_trials  # thid dim: n_trials = n_neurons (all stimulated ad once)

    if debug_plot:
        for i in range(input_current.shape[2]):
            plt.plot(t, input_current[0, :, i])
        plt.xlabel('Time (sec)')
        plt.ylabel('Input current')
        plt.gcf().savefig('./debug_get_input_current_oscillation.pdf')

    return input_current, list_mean_current


def sweep_frequency_oscillations(frequencies=np.arange(20, 30), n_trials=10, offset=10, amplitude_100=5, fs=1000,
                                 target_snr_db=20, debug_plot=True, add_noise=True, stim_length_sec=1):
    """
    Return amplitude of input current across time, with as many input signals as the dimension of
    the input amplitudes.

    dt_sec:
    stim_length_sec:
    amplitudes
    """

    n_neurons = len(frequencies) * n_trials
    stim = []
    area = amplitude_100 / np.pi / 100 / 2
    list_mean_frequency = []  # list with mean current value (same dimension as the last dimension of input_current)
    for f in frequencies:
        t = np.arange(fs * stim_length_sec) / fs
        n_time_bins = len(t)
        amplitude = area * np.pi * f * 2
        for n in range(n_trials):
            x = amplitude * np.sin(2 * np.pi * f * t) + offset
            x_watts = x ** 2

            sig_avg_watts = np.mean(x_watts)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            # Calculate noise according to [2] then convert to watts
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            # Generate an sample of white noise with mean zero. For white noise, Ex and the average power is then equal to the variance Ex.
            mean_noise = 0
            noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
            # Noise up the original signal
            I = x + noise_volts * add_noise

            stim.append(torch.tensor(I))
            list_mean_frequency.append(f)

    input_current = torch.stack(stim, dim=1)
    input_current = torch.reshape(input_current, (n_time_bins, n_neurons))
    input_current = input_current[None, :]  # add first dimension for batch size 1

    assert input_current.shape[0] == 1
    assert input_current.shape[1] == n_time_bins
    assert input_current.shape[2] == len(
        frequencies) * n_trials  # thid dim: n_trials = n_neurons (all stimulated ad once)

    if debug_plot:
        for i in range(input_current.shape[2]):
            plt.plot(t, input_current[0, :, i])
        plt.xlabel('Time (sec)')
        plt.ylabel('Input current')
        plt.gcf().savefig('./debug_get_input_current_oscillation.pdf')
    unique_freqs = np.unique(list_mean_frequency)
    list_mean_frequency_index = []
    for mean_frequency in list_mean_frequency:
        list_mean_frequency_index.append(np.where(unique_freqs == mean_frequency)[0][0])
    return input_current, list_mean_frequency_index


def train_nmf_histogram_isi(dl_4nmf, neuron, params, device, rank_NMF, writer):
    H_save = []
    Y_save = []
    V_save = []

    for x_local, y_local in dl_4nmf:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        params['nb_channels'] = x_local.shape[0]
        neuron.N = params['nb_channels']
        neuron.reset()

        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(
                x_local[None, :, t])  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        indexes = torch.argsort(y_local)
        aaa = torch.where(s_out_rec_train[0, :, indexes])
        # plt.scatter(aaa[0].cpu(), aaa[1].cpu(),)
        # plt.show()
        # s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))
        # plt.figure()
        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        count_spikes = torch.sum(s_out_rec_train, dim=1)
        bins = 1000
        hist = torch.zeros(bins, s_out_rec_train.shape[1])
        # plt.figure()
        for i in range(s_out_rec_train.shape[1]):
            aaa = torch.diff(torch.where(s_out_rec_train[:, i])[0])
            # plt.plot(aaa.cpu())
            hist[:, i] = torch.histogram(aaa.to(torch.float), bins=bins)[0]
        s_out_rec_train = hist
        # plt.figure()
        # plt.plot(s_out_rec_train.cpu())
        # plt.show()

        # H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        V_matrix = s_out_rec_train.T
        V_save.append(V_matrix)
        Y_save.append(y_local)
    V_save = torch.vstack(V_save)
    net = NMF(V_save.shape, rank=rank_NMF)

    net.fit(V_save)
    # H_save.append(torch.tensor(net.H))
    # print(net.H)
    # H_save = torch.vstack(H_save)
    Y_save = torch.concat(Y_save)
    # Y_eee = torch.argsort(Y_save)
    # Y_save_sort = Y_save[Y_eee]
    H_save = net.H.clone().detach()
    # plt.imshow(H[Y_eee,:].cpu(), aspect='auto', interpolation='none')
    # plt.figure()
    # plt.imshow(V_save[Y_eee,:].cpu(), aspect='auto', interpolation='none')
    # plt.show()

    return H_save, Y_save, V_save, count_spikes


def train_nmf_histogram_count(dl_4nmf, neuron, params, device, rank_NMF, writer):
    H_save = []
    Y_save = []
    V_save = []

    for x_local, y_local in dl_4nmf:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        params['nb_channels'] = x_local.shape[0]
        neuron.N = params['nb_channels']
        neuron.reset()

        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(
                x_local[None, :, t])  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        indexes = torch.argsort(y_local)
        # aaa = torch.where(s_out_rec_train[0,:,indexes])

        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        count_spikes = torch.sum(s_out_rec_train, dim=1)
        bins = 1000
        hist = torch.zeros(bins, s_out_rec_train.shape[1])
        for i in range(s_out_rec_train.shape[1]):
            hist[:, i] = torch.histogram(s_out_rec_train[:, i], bins=bins)[0]
        s_out_rec_train = hist
        # plt.figure()
        # plt.plot(s_out_rec_train.cpu())
        # plt.show()
        # H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        V_matrix = s_out_rec_train.T
        V_save.append(V_matrix)
        Y_save.append(y_local)
    V_save = torch.vstack(V_save)
    net = NMF(V_save.shape, rank=rank_NMF)

    net.fit(V_save)
    # H_save.append(torch.tensor(net.H))
    # print(net.H)
    # H_save = torch.vstack(H_save)
    Y_save = torch.concat(Y_save)
    # Y_eee = torch.argsort(Y_save)
    # Y_save_sort = Y_save[Y_eee]
    H_save = net.H.clone().detach()
    # plt.imshow(H[Y_eee,:].cpu(), aspect='auto', interpolation='none')
    # plt.figure()
    # plt.imshow(V_save[Y_eee,:].cpu(), aspect='auto', interpolation='none')
    # plt.show()

    return H_save, Y_save, V_save, count_spikes


def train_nmf(dl_4nmf, neuron, params, device, rank_NMF, writer):
    H_save = []
    Y_save = []
    V_save = []

    for x_local, y_local in dl_4nmf:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        params['nb_channels'] = x_local.shape[0]
        neuron.N = params['nb_channels']
        neuron.reset()

        s_out_rec = []
        mem_rec = []
        thr = []
        b_dec = []
        b_update = []
        for t in range(x_local.shape[1]):
            out = neuron(
                x_local[None, :, t])  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
            mem_rec.append(neuron.new_mem)
            thr.append(neuron.thr)
            b_update.append(neuron.b_update)
            b_dec.append(neuron.b_dec)
        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        mem_rec = torch.stack(mem_rec, dim=1)
        thr = torch.stack(thr, dim=1)
        b_update = torch.stack(b_update, dim=1)
        b_dec = torch.stack(b_dec, dim=1)
        # indexes = torch.argsort(y_local)
        # aaa = torch.where(s_out_rec_train[0,:,indexes])
        # plt.scatter(aaa[0].cpu(), aaa[1].cpu(),)
        # plt.show()
        # s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))

        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        mem_rec = torch.flatten(mem_rec, start_dim=0, end_dim=1)
        thr = torch.flatten(thr, start_dim=0, end_dim=1)
        b_update = torch.flatten(b_update, start_dim=0, end_dim=1)
        b_dec = torch.flatten(b_dec, start_dim=0, end_dim=1)
        count_spikes = torch.sum(s_out_rec_train, dim=1)

        # H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        V_matrix = s_out_rec_train.T
        V_save.append(V_matrix)
        Y_save.append(y_local)

        fig, axs = plt.subplots(4, 1, sharex=True)
        trial_id = 0
        axs[0].plot(x_local[trial_id, :], label='input')
        axs[0].set_title(f'ro={np.round(neuron.ro,2)} - beta_adapt={neuron.beta_adapt}')
        axs[0].legend()
        axs[1].plot(mem_rec[:, trial_id], label='Vmem')
        axs[1].plot(thr[:, trial_id], label='thr')
        ax2 = axs[1].twinx()
        ax2.plot(b_update[:, trial_id], '.-', color='grey', label='@ sp')
        ax2.plot(b_dec[:, trial_id], '.-', color='k', label='decay')
        axs[1].legend()
        spike_times = np.where(V_matrix[trial_id, :] > 0)[0]
        axs[2].vlines(spike_times, 0, 1)
        axs[3].plot(spike_times[1:], np.diff(spike_times), '.-', label='isi')
        axs[3].legend()
        axs[3].set_xlabel('Time (ms)')
        plt.show()

    V_save = torch.vstack(V_save)
    plt.figure()
    plt.imshow(V_save)

    net = NMF(V_save.shape, rank=rank_NMF)

    net.fit(V_save)
    # H_save.append(torch.tensor(net.H))
    # print(net.H)
    # H_save = torch.vstack(H_save)
    Y_save = torch.concat(Y_save)
    # Y_eee = torch.argsort(Y_save)
    # Y_save_sort = Y_save[Y_eee]
    H_save = net.H.clone().detach()
    # plt.imshow(H[Y_eee,:].cpu(), aspect='auto', interpolation='none')
    # plt.figure()
    # plt.imshow(V_save[Y_eee,:].cpu(), aspect='auto', interpolation='none')
    # plt.show()

    return H_save, Y_save, V_save, count_spikes


def train(dl_train, neuron, params, device, rank_NMF, optimizer, model, criterion, writer, epoch):
    list_epoch_loss = []
    for x_local, y_local in dl_train:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        # params['nb_channels'] = x_local.shape[0]
        # neuron.N = params['nb_channels']
        # neuron.reset()
        #
        # s_out_rec = []
        # for t in range(x_local.shape[1]):
        #     out = neuron(
        #         x_local[None,:, t])  # shape: n_batches x 1 fanout x n_param_values x n_channels
        #     s_out_rec.append(out)
        # s_out_rec_train = torch.stack(s_out_rec, dim=1)
        # # indexes = torch.argsort(y_local)
        # # aaa = torch.where(s_out_rec_train[0,:,indexes])
        # # plt.scatter(aaa[0].cpu(), aaa[1].cpu(),)
        # # plt.show()
        # # s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))
        #
        # ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        # s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        # # H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        # V_matrix = s_out_rec_train.T
        # net = NMF(V_matrix.shape, rank=rank_NMF)
        # net.fit(V_matrix)
        # # H[0] = net.H
        # eee = torch.argsort(y_local)
        # plt.imshow(net.H[eee,:].detach().cpu(), aspect='auto')
        # plt.show()
        optimizer.zero_grad()
        outputs = model(x_local)
        # outputs_norm = (outputs - label_min) / (label_diff)
        loss = criterion(outputs[:, 0], y_local)
        # get gradients w.r.t to parameters
        loss.backward()
        list_epoch_loss.append(loss.item())
        # update parameters
        optimizer.step()
    writer.add_scalar('MSE/train', torch.mean(torch.tensor(list_epoch_loss)).cpu().numpy(), epoch)
    return torch.mean(torch.tensor(list_epoch_loss)).clone().cpu().numpy()
    # print('Epoch: {}. MSE train: {}. '.format(e, loss.item()))


def eval(dl_test, neuron, params, device, rank_NMF, model, criterion, writer, epoch):
    list_epoch_loss_test = []
    list_epoch_MI = []
    with torch.no_grad():
        for x_local, y_local in dl_test:
            predicted = model(x_local)
            loss_test = criterion(predicted[:, 0], y_local)
            #
            label = y_local
            label_unique = torch.unique(label)
            predicted_int = predicted.type(torch.int)
            predicted_range = torch.unique(predicted_int)
            # print(predicted_range)
            pdf_x1x2 = torch.zeros([len(label_unique), len(predicted_range)])
            # print(predicted_range)
            for trial_idx in range(len(predicted_int)):
                lab_pos = torch.where(label_unique == label[trial_idx])[0]
                pred_pos = torch.where(predicted_range == predicted_int[trial_idx])[0]
                pdf_x1x2[lab_pos, pred_pos] += 1
            num_occ = torch.sum(pdf_x1x2)

            # plt.show()
            pdf_x1 = torch.sum(pdf_x1x2, dim=1) / num_occ  # to check
            pdf_x2 = torch.sum(pdf_x1x2, dim=0) / num_occ
            pdf_x1x2 = pdf_x1x2 / num_occ
            mi = torch.zeros(1)
            for el1_idx, pdf_x1_el in enumerate(pdf_x1):
                for el2_idx, pdf_x2_el in enumerate(pdf_x2):
                    mi += pdf_x1x2[el1_idx, el2_idx] * torch.log2(
                        (pdf_x1x2[el1_idx, el2_idx] / (pdf_x1_el * pdf_x2_el)) + 1E-10)
            list_epoch_MI.append(mi.item())
            list_epoch_loss_test.append(loss_test.item())
        # print('Epoch: {}. MSE test: {}. '.format(epoch, mi.item()))

        fig1, axis1 = plt.subplots()
        fig2, axis2 = plt.subplots()
        pcm = axis1.imshow([predicted.cpu().numpy(), label[:, None].cpu().numpy()], aspect='auto')
        # axis1.colorbar()
        fig1.colorbar(pcm, ax=axis1)
        pcm2 = axis2.imshow(pdf_x1x2.cpu().numpy(), aspect='auto')
        fig2.colorbar(pcm2, ax=axis2)
        eee = torch.argsort(y_local)
        # aaa = torch.where(s_out_rec_test[:,eee])
        # axis3.scatter(aaa[1].cpu().numpy(),aaa[0].cpu().numpy())
        writer.add_figure(figure=fig1, global_step=epoch, tag='test_resp')
        writer.add_figure(figure=fig2, global_step=epoch, tag='test_mi')
        writer.add_scalar('MI', torch.mean(torch.tensor(list_epoch_MI)).cpu().numpy(), epoch)
        # writer.add_scalar('MSE/train', torch.mean(torch.tensor(list_epoch_loss)).cpu().numpy(), e)
        writer.add_scalar('MSE/test', torch.mean(torch.tensor(list_epoch_loss_test)).cpu().numpy(), epoch)

    return torch.mean(torch.tensor(list_epoch_MI)).cpu().numpy(), torch.mean(
        torch.tensor(list_epoch_loss_test)).cpu().numpy()


def LIF_neuron_params(neuron_param_values, name_param_sweeped, extremes_sweep, MNclass, data, labels, dt_sec, name):
    # Set results folder:
    iscuda = torch.cuda.is_available()
    # # Forcing CPU
    iscuda = False
    # print('NOTE: Forced CPU')

    if run_with_fake_input:
        results_dir = 'results/controlled_stim/'
        if not (os.path.isdir(results_dir)):
            os.mkdir(results_dir)
    else:
        results_dir = set_results_folder([exp_id, name, MNclass, name_param_sweeped, str(extremes_sweep)])
        results_dir += '/'

    # Filename metadata:
    metadatafilename = results_dir + '/metadata.txt'

    # Create file with metadata
    addHeaderToMetadata(metadatafilename, 'Simulation')

    device = torch.device(
        'cuda') if iscuda else torch.device('cpu')
    torch.manual_seed(0)

    cuda = iscuda
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    ###########################################
    ##                Dataset                ##
    ###########################################

    x_train, x_test, y_train, y_test = train_test_split(
        data.cpu(), labels, test_size=0.2, shuffle=True, stratify=labels)

    ds_train = TensorDataset(x_train, torch.tensor(y_train).to_dense().to(torch.float32))
    ds_test = TensorDataset(x_test, torch.tensor(y_test).to_dense().to(torch.float32))
    ds_4nmf = TensorDataset(data.cpu(),
                            torch.tensor(labels).to_dense().to(torch.float32))  # data: n trials x n timestamps
    params['nb_channels'] = 1
    params['labels'] = labels

    params['nb_channels'] = 1
    params['data_steps'] = dt_sec

    # Network parameters
    # Learning parameters
    nb_epochs = int(50)
    ###########################################
    ##                Network                ##
    ###########################################

    n_param_values = params['n_param_values']

    tau_mem = 0.01  # sec
    tau_ratio = 2
    tau_syn = tau_mem / tau_ratio
    alpha = float(np.exp(-dt_sec / tau_syn))
    beta = float(np.exp(-dt_sec / tau_mem))
    # #n_inputs = data.shape[0]
    # n_outputs = n_inputs
    tau_adp = 2# 0.5

    batch_size = int(100)

    neuron = models.ALIF_neuron(batch_size,
                                batch_size,
                                alpha,
                                beta,
                                is_recurrent=False,
                                fwd_weight_scale=1.0,
                                rec_weight_scale=1.0,
                                b_0=neuron_param_values['b_0'],
                                dt=dt_sec,
                                tau_adp=tau_adp,
                                beta_adapt=neuron_param_values['beta_adapt'],
                                analog_input=True)

    # inputDim = 1  # takes variable 'x'
    # outputDim = 1  # takes variable 'y'
    learningRate = params['learningRate']  # 10#0.01

    # neuron_id = int(params['neuron_id'])

    rank_NMF = int(params['rank_NMF'])
    model = linearRegression(rank_NMF, 1)

    # TODO: Check this
    if iscuda:
        pin_memory = True
        num_workers = 0
    else:
        pin_memory = False
        num_workers = 0

    # The log softmax function across output units
    # dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    #                       generator=torch.Generator(device=device))
    # dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    #                      generator=torch.Generator(device=device))
    dl_4nmf = DataLoader(ds_4nmf, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         generator=torch.Generator(device=device))
    # pbar = trange(nb_epochs)
    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    optimizer = torch.optim.Adamax(model.parameters(), lr=learningRate)
    params['optimizer'] = optimizer.__class__.__name__
    # print('Training classifier with {} optimizer'.format(params['optimizer']))

    # Store parameters to metadata file:
    for key in params.keys():
        addToNetMetadata(metadatafilename, key, params[key])

    header = 'Neuron params'
    for key in neuron_param_values.keys():
        addToNetMetadata(metadatafilename, key, neuron_param_values[key], header=header)
        header = ''
    writer = SummaryWriter(
        comment="Name" + name + "Stim" + str(name_param_sweeped) + "_Range" + str(extremes_sweep) + "_MN_class" + str(
            MNclass))
    list_loss = []
    list_mi = []
    list_mse_test = []
    pbar = trange(nb_epochs)

    if name == 'count':
        nmf_samples, nmf_labels, V_save, spike_count = train_nmf_histogram_count(dl_4nmf, neuron, params, device,
                                                                                 rank_NMF, writer)
    elif name == 'isi':
        nmf_samples, nmf_labels, V_save, spike_count = train_nmf_histogram_isi(dl_4nmf, neuron, params, device,
                                                                               rank_NMF, writer)
    elif name == 'spike':
        nmf_samples, nmf_labels, V_save, spike_count = train_nmf(dl_4nmf, neuron, params, device,
                                                                 rank_NMF, writer)
    else:
        raise ValueError('Wrong name for the encoding')
    # print('we')
    h = list(LIFclasses.keys()).index(MNclass)
    eee = torch.argsort(nmf_labels)
    writer.add_histogram('Spike Count', spike_count, bins='auto', global_step=h)
    ax4[h].imshow(nmf_samples[eee, :], aspect='auto', cmap='Greens', interpolation='none')
    ax4[h].set_title(MNclass)

    x_train, x_test, y_train, y_test = train_test_split(
        nmf_samples.cpu(), nmf_labels, test_size=0.2, shuffle=True, stratify=labels)
    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          generator=torch.Generator(device=device))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         generator=torch.Generator(device=device))
    fig1, axis1 = plt.subplots(1, 1)
    eee = torch.argsort(nmf_labels)
    aaa = torch.where(V_save[eee, :])
    plt.imshow(V_save[eee, :])
    plt.show()
    axis1.scatter(aaa[1].cpu().numpy(), aaa[0].cpu().numpy())
    h = list(LIFclasses.keys()).index(MNclass)
    ax3[h].scatter(aaa[1].cpu().numpy(), aaa[0].cpu().numpy())
    ax3[h].set_title(MNclass)
    # writer.add_figure(figure=fig4, global_step=0, tag='test_nmf')
    # writer.add_figure(figure=fig3, global_step=0, tag='test_V')
    writer.add_figure('V', fig1, 0)

    for e in range(nb_epochs):
        list_epoch_loss = []
        list_epoch_MI = []
        list_epoch_loss_test = []
        loss_train = train(epoch=e, model=model, optimizer=optimizer, criterion=criterion, dl_train=dl_train,
                           device=device, rank_NMF=rank_NMF, writer=writer, neuron=neuron, params=params)
        list_loss.append(loss_train.item())

        # get gradients w.r.t to parameters
        # update parameters
        mi_test, mse_test = eval(epoch=e, model=model, criterion=criterion, dl_test=dl_test, device=device,
                                 rank_NMF=rank_NMF, writer=writer, neuron=neuron, params=params)
        list_mi.append(mi_test.item())
        list_mse_test.append(mse_test.item())
        # list_mi.append(torch.mean(torch.tensor(list_epoch_mi)).cpu().numpy())
        # print('plotting')
        # fig1,axis1 = plt.subplots()
        # plt.figure()
        # axis1.plot(list_loss,color='b')
        # axis2 = axis1.twinx()
        # axis2.plot(list_mi, color='r')
        # plt.show()
    # print('Saving results')
    torch.save(list_loss, results_dir + 'Loss.pt')
    torch.save(list_mi, results_dir + 'MI.pt')
    torch.save(list_mse_test, results_dir + 'MSE.pt')
    torch.save(spike_count, results_dir + 'Spike_Count.pt')


from multiprocessing import Process


def run_in_parallel(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


def calculate_LIF_class(stimuli_type, range_val, LIFclass, data, labels, dt_sec, encoding_method, sema=None):
    print('-------------------------------------------')

    # Run mutual information analysis
    dict_keys = LIFclasses[LIFclass]
    LIF_neuron_params(dict_keys, stimuli_type, range_val, LIFclass, data, labels, dt_sec=dt_sec, name=encoding_method)

    print('-------------------------------------------')
    print('DONE. Class {}, Stimuli {}, Range {}, Encoding {}'.format(LIFclass, stimuli_type,
                                                                     [range_val[0], range_val[-1]],
                                                                     encoding_method))
    if sema is not None:
        sema.release()


from multiprocessing import Semaphore, Process

if __name__ == "__main__":
    name = 'count'
    print('Current path', Current_PATH)
    n_trials = 100
    last = 5
    stim_length_sec = 10
    noise = 0.1
    dt_sec = 0.001
    debug_plot = False

    for stimuli_type in ranges.keys():
        for range_val in ranges[stimuli_type]:
            # upsample_fac = 5
            n_time_bins = int(np.floor(stim_length_sec / dt_sec))
            # amplitudes = np.linspace(1, 10, 10)
            if stimuli_type == 'ampli':
                data, labels = sweep_steps(amplitudes=range_val, n_trials=n_trials, dt_sec=dt_sec,
                                           stim_length_sec=stim_length_sec, sig=noise, debug_plot=debug_plot)
            elif stimuli_type == 'ampli_neg':
                data, labels = sweep_steps(amplitudes=range_val, n_trials=n_trials, dt_sec=dt_sec,
                                           stim_length_sec=stim_length_sec, sig=noise, debug_plot=debug_plot)
                data = -data
            elif stimuli_type == 'freq':
                data, labels = sweep_frequency_oscillations(frequencies=range_val, n_trials=n_trials,
                                                            offset=0, amplitude_100=4, fs=1 / dt_sec, target_snr_db=20,
                                                            debug_plot=debug_plot, add_noise=noise > 0)
            elif stimuli_type == 'freq_pos':
                data, labels = sweep_frequency_oscillations(frequencies=range_val, n_trials=n_trials,
                                                            offset=0, amplitude_100=4, fs=1 / dt_sec,
                                                            target_snr_db=20,
                                                            debug_plot=debug_plot, add_noise=noise > 0)
                data[data < 0] = 0

            elif stimuli_type == 'freq_neg':
                data, labels = sweep_frequency_oscillations(frequencies=range_val, n_trials=n_trials,
                                                            offset=0, amplitude_100=4, fs=1 / dt_sec,
                                                            target_snr_db=20,
                                                            debug_plot=debug_plot, add_noise=noise > 0)
                data[data < 0] = 0
                data = -data
            elif stimuli_type == 'slopes':
                data, labels = sweep_slopes(slopes=range_val, n_trials=n_trials, dt_sec=dt_sec,
                                            stim_length_sec=stim_length_sec,
                                            last=last, first=0, sig=noise, debug_plot=debug_plot)
            data = data[0, :, :].T  # n neurons (number of total input values) x (n time stamps)
            # plt.plot(data.T)
            for encoding_method in encoding_methods:
                if multiprocess:
                    concurrency = 8
                    sema = Semaphore(concurrency)
                    all_processes = []
                    for LIFclass in LIFclasses:
                        sema.acquire()
                        p = Process(target=calculate_LIF_class, args=(stimuli_type, range_val, LIFclass, data, labels,
                                                                      dt_sec, encoding_method, sema))
                        all_processes.append(p)
                        p.start()
                else:
                    for LIFclass in LIFclasses:
                        print(f'Class {LIFclass}')
                        calculate_LIF_class(stimuli_type, range_val, LIFclass, data, labels, dt_sec, encoding_method,
                                            None)
                # for MNclass in MNclasses:
                #     print('-------------------------------------------')
                #     print('Class {}, Stimuli {}, Range {}, Encoding {}'.format(MNclass,stimuli_type,[range_val[0],range_val[-1]],encoding_method))
                #     # Generate dictionary with parameter values:
                #     dict_keys = generate_dict('a', [MNclasses[MNclass]['a']],force_param_dict=MNclasses[MNclass])
                #     # Run mutual information analysis
                #     MI_neuron_params(dict_keys, stimuli_type, range_val, MNclass,data,labels,dt_sec=dt_sec,name= encoding_method)

            # Run mutual information analysis
            # MI_neuron_params(dict_keys, 'ampli_neg', [dict_keys['a'],dict_keys['a']], MNclass,data_ampli_neg,labels_ampli,dt_sec=dt_sec)

            # Run mutual information analysis
            # MI_neuron_params(dict_keys, 'freqs', [dict_keys['a'], dict_keys['a']], MNclass, data_freqs, labels_freq, dt_sec=dt_sec)

            # Run mutual information analysis
            # MI_neuron_params(dict_keys, 'slopes', [dict_keys['a'], dict_keys['a']], MNclass, data_slope, labels_slope, dt_sec=dt_sec)

plt.show()
