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
import argparse
from pathlib import Path

from datasets import load_analog_data
import time
import pickle
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
from sklearn.model_selection import train_test_split

Current_PATH = os.getcwd()
# matplotlib.use('Agg')
# Set seed:
seed = 6
torch.manual_seed(seed)
np.random.seed(seed)

# ---------------------------- Input -----------------------------------
save_out = True  # Flag to save figures:
sweep_param_name = ['a', 'A1', 'A2', 'b', 'G', 'k1', 'k2']
sweep_ranges = [[-10, 10], [-100, 100], [-1000, 1000]]

MNclasses = {
    'Tonic': {'a':0,'A1':0,'A2': 0,"b" :10,"G" : 50,"k1" :  200,"k2" : 20,"gain" : 1, 'R1': 0, 'R2': 1},
    'Adaptive': {'a':5,'A1':0,'A2': 0,"b" :10,"G" : 50,"k1" :  200,"k2" : 20,"gain" : 1, 'R1': 0, 'R2': 1},
    # 'K': {'a':30,'A1':0,'A2': 0,"b" :10,"G" : 50,"k1" :  200,"k2" : 20,"gain" : 1, 'R1': 0, 'R2': 1},
    # 'L': {'a':30,'A1':10,'A2': -0.6,"b" :10,"G" : 50,"k1" :  200,"k2" : 20,"gain" : 1, 'R1': 0, 'R2': 1},
    # 'M2O': {'a':5,'A1':10,'A2': -0.6,"b" :10,"G" : 50,"k1" :  200,"k2" : 20,"gain" : 1, 'R1': 0, 'R2': 1},
    # 'P2Q': {'a':5,'A1':5,'A2': -0.3,"b" :10,"G" : 50,"k1" :  200,"k2" : 20,"gain" : 1, 'R1': 0, 'R2': 1},
    #'R': {'a':0,'A1':8,'A2': -0.1,"b" :10,"G" : 50,"k1" :  200,"k2" : 20,"gain" : 1, 'R1': 0, 'R2': 1},
    # 'S': {'a':5,'A1':-3,'A2': 0.5,"b" :10,"G" : 50,"k1" :  200,"k2" : 20,"gain" : 1, 'R1': 0, 'R2': 1},
    # 'T': {'a':-80,'A1':0,'A2': 0,"b" :10,"G" : 50,"k1" :  200,"k2" : 20,"gain" : 1, 'R1': 0, 'R2': 1},
    # 'opt': {'a':2.743,'A1':0.03712,'A2': -0.5089,"b" :11.4,"G" : 47.02,"k1" :  200,"k2" : 20,"gain" : 1, 'R1': 0, 'R2': 1},
   # 'Braille':{"A1": -0.015625353902578354, "G": 45.24007797241211, "a": 2.6239240169525146, "A2": -1.0590057373046875, "k2": 20.0, "b": 12.77495288848877, "R2": -1.1421641111373901, "R1": 0.3858567178249359, "k1": 200.0}
}

# default_ranges {
#
# }
# ranges = {
#     'ampli':[np.linspace(1, 6, 10)],#,np.linspace(1, 100, 10),np.linspace(1, 1000, 10)],
#     'freq_pos':[np.linspace(10,200,10)],#,np.linspace(10,500,10),np.linspace(10,1000,10)],
#     # 'freq_neg': [np.linspace(10, 100, 10), np.linspace(10, 500, 10), np.linspace(10, 1000, 10)],
#     # 'ampli_neg':[np.linspace(1, 4, 10),np.linspace(1, 10, 10),np.linspace(1, 100, 10)],
#     'slopes':[np.linspace(1, 0.05, 10)]
#     #     # np.linspace(1, 0.4, 10),
#     #     np.linspace(1, 0.05, 10)
#     # ]
#
# }
encoding_methods = [
    'spike',
    # 'count',
    # 'isi',
    # 'isi_nonmf'
]
# encoding_methods = ['count']
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

# fig3,ax3 = plt.subplots(len(MNclasses)+1*int(len(MNclasses) == 1),1)
# fig4,ax4 = plt.subplots(len(MNclasses)+1*int(len(MNclasses) == 1),1)
# fig5,ax5 = plt.subplots()

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


def sweep_slopes(slopes=np.arange(10), n_trials=10, dt_sec=0.001, stim_length_sec=0.1, sig=.1, debug_plot=True,last=5,first = 0):
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
    slopes = slopes/dt_sec
    # list with mean current value (same dimension as the last dimension of input_current)
    for a in slopes:
        time_necessary = (last - first)/a
        time_first = (stim_length_sec - time_necessary)/2
        time_last = (stim_length_sec - time_necessary)/2
        assert time_first > dt_sec
        assert time_necessary > dt_sec
        first_vec = np.linspace(first, first, int(np.floor(time_first / dt_sec)))
        last_vec = np.linspace(last, last, int(np.floor(time_last / dt_sec)))
        slope = np.linspace(0,last-first,int(np.floor(time_necessary/dt_sec)))
        # slope = np.arange(0,last-first,time_necessary/dt_sec)
        for n in range(n_trials):
            # stim.append(torch.tensor([a] * n_time_bins))
            try:

                value = np.concatenate([first_vec,slope,last_vec])
                remaining = np.linspace(last, last,n_time_bins - len(value))
                value = np.concatenate([value,remaining])
                I_gwn =  value + sig * np.random.randn(n_time_bins) / np.sqrt(n_time_bins / 1000.)
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
    np.random.seed(seed)
    n_neurons = len(frequencies) * n_trials
    stim = []
    area = amplitude_100/np.pi/100
    list_mean_frequency = []  # list with mean current value (same dimension as the last dimension of input_current)
    for f in frequencies:
        t = np.arange(int(fs*stim_length_sec)) / fs
        n_time_bins = len(t)
        amplitude = area*np.pi*f
        amplitude=10
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
# def train_nmf_histogram_isi(V_save,Y_save,rank_NMF):
#     net = NMF(V_save.shape, rank=rank_NMF)
#     net.fit(V_save)
#     Y_save = torch.concat(Y_save)
#     H_save = net.H.clone().detach()
#     return H_save, Y_save
# def train_nmf_histogram_count(V_save,Y_save,rank_NMF):
#     net = NMF(V_save.shape, rank=rank_NMF)
#     net.fit(V_save)
#     Y_save = torch.concat(Y_save)
#     H_save = net.H.clone().detach()
#     return H_save, Y_save

def train_nmf(V_save, Y_save,rank_NMF):
    net = NMF(V_save.shape, rank=rank_NMF)
    net.fit(V_save,verbose=False)
    Y_save = torch.concat(Y_save)
    H_save = net.H.clone().detach()
    return H_save,Y_save

def train(dl_train, neuron, params, device,rank_NMF,optimizer,model,criterion,writer,epoch,name=''):
    list_epoch_loss = []
    for x_local, y_local in dl_train:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(x_local)
        loss = criterion(outputs, y_local)
        loss.backward()
        list_epoch_loss.append(loss.item())
        optimizer.step()
    writer.add_scalar('Loss/train', torch.mean(torch.tensor(list_epoch_loss)).clone().cpu().numpy(), epoch)
    return torch.mean(torch.tensor(list_epoch_loss)).clone().cpu().numpy()

def eval(dl_test, neuron, params, device,rank_NMF,model,criterion,writer,epoch,name=''):

    list_epoch_loss_test = []
    list_epoch_MI = []
    list_epoch_accuracy = []
    with torch.no_grad():
        for x_local, y_local in dl_test:
            predicted = model(x_local)
            loss_test = criterion(predicted, y_local)
            label = y_local
            label_unique = torch.unique(label)
            predicted_int = torch.argmax(predicted,dim=1)
            predicted_range = torch.unique(predicted_int)
            pdf_x1x2 = torch.zeros([len(label_unique), len(predicted_range)])
            for trial_idx in range(len(predicted_int)):
                lab_pos = torch.where(label_unique == label[trial_idx])[0]
                pred_pos = torch.where(predicted_range == predicted_int[trial_idx])[0]
                pdf_x1x2[lab_pos, pred_pos] += 1
            num_occ = torch.sum(pdf_x1x2)
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
            list_epoch_accuracy.append(torch.sum(predicted_int == label).item() / len(label))
            writer.add_scalar('MI', torch.mean(torch.tensor(list_epoch_MI)).cpu().numpy(), epoch)
            writer.add_scalar('Accuracy_test', torch.mean(torch.tensor(list_epoch_accuracy)).cpu().numpy(), epoch)
            writer.add_scalar('Loss/test', torch.mean(torch.tensor(list_epoch_loss_test)).cpu().numpy(), epoch)
    return torch.mean(torch.tensor(list_epoch_MI)).cpu().numpy(), torch.mean(torch.tensor(list_epoch_loss_test)).cpu().numpy(), torch.mean(torch.tensor(list_epoch_accuracy)).cpu().numpy()
def MI_neuron_params(neuron_param_values, name_param_sweeped, extremes_sweep, MNclass,data,labels,dt_sec,name,writer,args):

    iscuda = (torch.cuda.is_available()) and args.gpu
    device = torch.device(
        'cuda') if iscuda else torch.device('cpu')
    torch.manual_seed(args.seed)

    cuda = iscuda
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    ###########################################
    ##                Dataset                ##
    ###########################################
    label_u, labels_idx = np.unique(labels, return_inverse=True)
    label_n = len(label_u)
    x_train, x_test, y_train, y_test = train_test_split(
        data.cpu(), labels_idx, test_size=0.2, shuffle=True, stratify=labels, random_state=seed)
    ds_4nmf = TensorDataset(data.cpu(), torch.tensor(labels_idx).to_dense())
    params['nb_channels'] = 1
    params['labels'] = labels_idx
    params['data_steps'] = dt_sec

    # Network parameters
    # Learning parameters
    nb_epochs = int(args.n_epochs)
    ###########################################
    ##                Network                ##
    ###########################################
    tensor_params = dict.fromkeys(neuron_param_values.keys(), None)
    for key in tensor_params.keys():
        tensor_params[key] = torch.Tensor(neuron_param_values[key]).to(device)
    neuron = models.MN_neuron(1, {},
                                 a=tensor_params['a'],
                                 A1=tensor_params['A1'],
                                 A2=tensor_params['A2'],
                                 b=tensor_params['b'],
                                 G=tensor_params['G'],
                                 k1=tensor_params['k1'],
                                 k2=tensor_params['k2'],
                                 train=False,dt=dt_sec)

    batch_size = args.batch_size
    learningRate = args.lr

    # neuron_id = int(params['neuron_id'])

    rank_NMF = args.rank_NMF
    model = torch.nn.Linear(rank_NMF,label_n)

    # The log softmax function across output units
    # dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    #                       generator=torch.Generator(device=device))
    # dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    #                      generator=torch.Generator(device=device))
    dl_4nmf = DataLoader(ds_4nmf, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            generator=torch.Generator(device=device))
    # pbar = trange(nb_epochs)
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    optimizer = torch.optim.Adamax(model.parameters(), lr=learningRate)
    params['optimizer'] = optimizer.__class__.__name__
    # print('Training classifier with {} optimizer'.format(params['optimizer']))


    # writer = SummaryWriter(comment="Name" + name + "Stim" + str(name_param_sweeped) + "_c" + getattr(args, name_param_sweeped+'_center') + "_s" + getattr(args, name_param_sweeped+'_span') + "_MN_class"+str(MNclass))


    V_save_spike = []
    V_save_count = []
    V_save_isi = []
    Y_save = []
    V_save_isi_nonmf = []
    for x_local, y_local in dl_4nmf:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        params['nb_channels'] = x_local.shape[0]
        neuron.N = params['nb_channels']
        neuron.reset()

        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(
                x_local[None, :, t])
            s_out_rec.append(out)

        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        # spike_count = torch.sum(s_out_rec_train, dim=0)

        Y_save.append(y_local)
        V_matrix_spike = s_out_rec_train.T
        V_save_spike.append(V_matrix_spike)
        counter = 0
        count = torch.zeros(rank_NMF, s_out_rec_train.shape[1])
        for i in range(s_out_rec_train.shape[1]):
                count[:, i] = torch.histogram(s_out_rec_train[:, i], bins=torch.linspace(0,1,rank_NMF+1))[0]
                if y_local[i] == counter:

                    counter += 1
        V_matrix_count = count.T
        V_save_count.append(V_matrix_count)
        if 'isi' in args.encoding_methods:
            bins = s_out_rec_train.shape[0]
            hist = torch.zeros(bins, s_out_rec_train.shape[1])
            for i in range(s_out_rec_train.shape[1]):
                aaa = torch.diff(torch.where(s_out_rec_train[:, i])[0])
                hist[:,i] = torch.histogram(aaa.to(torch.float), bins=bins)[0]
            V_matrix_isi = hist.T
            V_save_isi.append(V_matrix_isi)
        if 'isi_nonmf' in args.encoding_methods:
            hist_nonmf = torch.zeros(rank_NMF, s_out_rec_train.shape[1])
            for i in range(s_out_rec_train.shape[1]):
                aaa = torch.diff(torch.where(s_out_rec_train[:, i])[0])
                hist_nonmf[:,i] = torch.histogram(aaa.to(torch.float), bins=rank_NMF)[0]
            V_matrix_isi_nonmf = hist_nonmf.T
            V_save_isi_nonmf.append(V_matrix_isi_nonmf)
    V_save_spike = torch.vstack(V_save_spike)
    V_save_count = torch.vstack(V_save_count)
    if 'isi' in args.encoding_methods:
        V_save_isi = torch.vstack(V_save_isi)
    if 'isi_nonmf' in args.encoding_methods:
        V_save_isi_nonmf = torch.vstack(V_save_isi_nonmf)
    for encoding_method in args.encoding_methods:
        name = os.path.join(name_param_sweeped, str(np.round(getattr(args,
                                                                     name_param_sweeped + '_center'), 2)) + str(
            np.round(getattr(args, name_param_sweeped + '_span'),
                     2)), encoding_method, MNclass,str(args.seed))
        if args.debug_plot:
            writer = SummaryWriter(log_dir='MN_DSP/runs/'+exp_id+"/"+name,comment=name)
        else:
            writer = SummaryWriter(log_dir='MN_DSP/runs/'+exp_id+"/"+name,comment=name)
        pbar = trange(nb_epochs)

        results_dir = set_results_folder([exp_id,MNclass, name_param_sweeped, str(np.round(getattr(args,
            name_param_sweeped + '_center'),2)) + str(np.round(getattr(args, name_param_sweeped + '_span'),
                                                               2)),encoding_method,str(args.seed)])
        results_dir += '/'
        # Filename metadata:
        metadatafilename = results_dir + '/metadata.txt'

        # Create file with metadata
        addHeaderToMetadata(metadatafilename, 'Simulation')
        # Store parameters to metadata file:
        for key in params.keys():
            addToNetMetadata(metadatafilename, key, params[key])

        header = 'Neuron params'
        for key in neuron_param_values.keys():
            addToNetMetadata(metadatafilename, key, neuron_param_values[key], header=header)
            header = ''
        if encoding_method == 'count':
            input_var = V_save_count
            labels = torch.hstack(Y_save)
            output_var = input_var
        elif encoding_method == 'isi':
            input_var = V_save_isi
            output_var, labels = train_nmf(V_save_isi,Y_save,rank_NMF)
        elif encoding_method == 'spike':
            input_var = V_save_spike
            output_var, labels = train_nmf(V_save_spike,Y_save,rank_NMF)
        elif encoding_method == 'isi_nonmf':
            input_var = V_save_isi_nonmf
            output_var, labels = train_nmf(V_save_isi_nonmf,Y_save,rank_NMF)
        else:
            raise ValueError('Wrong name for the encoding')
        if args.debug_plot:
            fig1, axis1 = plt.subplots(1, 1,figsize=(10,10))
            labels_ordered = torch.argsort(labels)
            axis1.imshow(input_var[labels_ordered, :].cpu().numpy(), aspect='auto',interpolation='none')

            axis1.set_title('Input '+name)
            fig2, axis2 = plt.subplots(1, 1,figsize=(10,10))
            axis2.imshow(output_var[labels_ordered,:].cpu().numpy(), aspect='auto',interpolation='none')
            axis2.set_title('Output '+name)
            writer.add_figure(figure=fig1, global_step=0, tag='Input')
            writer.add_figure(figure=fig2, global_step=0, tag='Output')

        x_train, x_test, y_train, y_test = train_test_split(
            output_var.cpu(), labels, test_size=0.2, shuffle=True, stratify=labels, random_state=seed)
        ds_train = TensorDataset(x_train, y_train)
        ds_test = TensorDataset(x_test, y_test)
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                                generator=torch.Generator(device=device))
        dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                                generator=torch.Generator(device=device))

        list_loss = []
        list_mi = []
        list_mse_test = []
        list_acc_test = []
        for e in pbar:
            loss_train = train(epoch=e, model=model, optimizer=optimizer, criterion=criterion, dl_train=dl_train, device=device,rank_NMF=rank_NMF, writer=writer,neuron=neuron,params=params,name=name)
            list_loss.append(loss_train.item())
            mi_test, mse_test,acc_test = eval(epoch=e, model=model, criterion=criterion, dl_test=dl_test, device=device,rank_NMF=rank_NMF, writer=writer,neuron=neuron,params=params,name=name)
            list_mi.append(mi_test.item())
            list_mse_test.append(mse_test.item())
            list_acc_test.append(acc_test.item())
            pbar.set_postfix_str('Loss: {:.4f}, MI: {:.4f}, MSE: {:.4f}, Acc: {:.4f}'.format(loss_train.item(), mi_test.item(), mse_test.item(), acc_test.item()))
        torch.save(list_loss, results_dir + 'Loss.pt')
        torch.save(list_mi, results_dir + 'MI.pt')
        torch.save(list_acc_test, results_dir + 'Accuracy.pt')
        torch.save(list_mse_test, results_dir + 'MSE.pt')
        torch.save(V_save_count, results_dir + 'Spike_Count.pt')
from multiprocessing import Process

def run_in_parallel(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

def calculate_MI_class(stimuli_type, range_val, MNclass,data, labels,dt_sec,name,writer,args):
    print('-------------------------------------------')
    print('Class {}, Stimuli {}, Range center {} span {}, Encoding {}, Seed {}'.format(MNclass, stimuli_type, np.round(getattr(args, stimuli_type + '_center'),2),
                                                                      np.round(getattr(args, stimuli_type + '_span'),
                                                                               2),name,args.seed))
    # Generate dictionary with parameter values:
    dict_keys = generate_dict('a', [MNclasses[MNclass]['a']], force_param_dict=MNclasses[MNclass])
    # Run mutual information analysis
    MI_neuron_params(dict_keys, stimuli_type, range_val, MNclass, data, labels, dt_sec=dt_sec, name=name,writer=writer,args = args)
    print('-------------------------------------------')
    print('DONE. Class {}, Stimuli {}, Range center {} span {}, Encoding {}, Seed {}'.format(MNclass, stimuli_type, np.round(getattr(args, stimuli_type + '_center'),2),
                                                                      np.round(getattr(args, stimuli_type + '_span'),
                                                                               2),name,args.seed))
    sema.release()

from multiprocessing import Semaphore,Process
if __name__ == "__main__":
    name = 'MN_DSP'
    parser = argparse.ArgumentParser(name)
    parser.add_argument('--name', type=str, default=name)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--last', type=int, default=5)
    parser.add_argument('--stim_length_sec', type=float, default=1.1)
    parser.add_argument('--noise', type=str, default='0.1,0,0')
    parser.add_argument('--dt_sec', type=float, default=0.001)
    parser.add_argument('--debug_plot', '-d', action='store_true')
    parser.add_argument('--load_range', type=str, default='Braille')
    parser.add_argument('--encoding_methods',type=str,default='spike')
    parser.add_argument('--load_neuron', type=str, default='')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--seed_n', type=int, default=100)
    parser.add_argument('--gpu',  action='store_true')
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--rank_NMF', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)



    ranges_possible = ['amplitude', 'amplitude_neg', 'frequency', 'frequency_neg', 'frequency_pos', 'slope']
    # ranges_possible = ['frequency']
    for range_name in ranges_possible:
        parser.add_argument(f'--{range_name}'+'_center', type=float, default=0)
        parser.add_argument(f'--{range_name}'+'_span', type=float, default=0)
        parser.add_argument(f'--{range_name}'+'_n_steps', type=int, default=0)


    args = parser.parse_args()

    if ',' in args.noise:
        args.noise = args.noise.split(',')
        args.noise = [float(noise) for noise in args.noise]
    else:
        args.noise = [float(args.noise)]
    if args.load_neuron == '':
        pass
    elif ',' in args.load_neuron:
        args.load_neuron = args.load_neuron.split(',')
        for MNclass in args.load_neuron:
            MNclasses[MNclass] = json.load(open('MN_params/' + MNclass + '.json'))
    else:
        MNclasses[args.load_neuron] = json.load(open('MN_params/' + args.load_neuron + '.json'))

    if ',' in args.encoding_methods:
        args.encoding_methods = args.encoding_methods.split(',')
    else:
        args.encoding_methods = [args.encoding_methods]
    # print('Current path',Current_PATH)
    folder_run = Path('dataset_analysis_hb')
    folder_stimuli = Path('stimuli')
    folder_run.mkdir(parents=True, exist_ok=True)
    folder_stimuli.mkdir(parents=True, exist_ok=True)
    folder_results = Path(f'experiments/results/{exp_id}')
    folder_results.mkdir(parents=True, exist_ok=True)
    ranges = {}
    stimuli_types = []
    if args.load_range == '':
        pass
    else:
        if (',' in args.load_range) == False:
            args.load_range = [args.load_range]
        elif ',' in args.load_range:
            args.load_range = args.load_range.split(',')

        for range_ds in args.load_range:
            json_range = json.load(open(f'{folder_run}/{range_ds}/data/opt.json'))
            for range_name in ranges_possible:
                try:
                    setattr(args, range_ds+'_'+range_name+'_center', json_range[range_name]['center'])
                    setattr(args, range_ds+'_'+range_name+'_span', json_range[range_name]['span'])
                    setattr(args, range_ds+'_'+range_name+'_n_steps', json_range[range_name]['n_steps'])

                    stimuli_types.append(range_name)
                    ranges[range_ds +'_'+ range_name] = [np.linspace(
                        getattr(args, range_ds + '_'+range_name + '_center') - getattr(args,
                                                                                   range_ds + '_'+range_name + '_span') / 2,
                        getattr(args, range_ds + '_'+range_name + '_center') + getattr(args,
                                                                                   range_ds + '_'+range_name + '_span') / 2,
                        getattr(args, range_ds + '_'+range_name + '_n_steps'))]
                except KeyError:
                    pass

    stimuli_types = np.unique(stimuli_types)
    if args.debug_plot:
        writer = SummaryWriter(log_dir='MN_DSP/runs')
    else:
        writer = None
    if args.seed == -1:
        seeds = range(args.seed_n)
    else:
        seeds = [args.seed]
        exp_id = 'Parallel'
    json.dump(args.__dict__, open(f'experiments/results/{exp_id}_args.json', 'w'))

    for seed_here in seeds:
        seed = seed_here
        args.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        for ds in args.load_range:
            for st_idx,stimuli_type in enumerate(stimuli_types):
                    stimuli = ds+'_'+stimuli_type
                    for range_val in ranges[stimuli]:
                        # upsample_fac = 5
                        n_time_bins = int(np.floor(args.stim_length_sec / args.dt_sec))
                        # amplitudes = np.linspace(1, 10, 10)
                        if stimuli_type == 'amplitude':
                            data, labels = sweep_steps(amplitudes=range_val, n_trials=args.n_trials, dt_sec=args.dt_sec,
                                                                   stim_length_sec=args.stim_length_sec, sig=args.noise[st_idx], debug_plot=args.debug_plot)
                        elif stimuli_type == 'amplitude_neg':
                            data, labels = sweep_steps(amplitudes=range_val, n_trials=args.n_trials, dt_sec=args.dt_sec,
                                                                   stim_length_sec=args.stim_length_sec, sig=args.noise[st_idx], debug_plot=args.debug_plot)
                            data = -data

                        elif stimuli_type == 'frequency':
                            data, labels = sweep_frequency_oscillations(frequencies=range_val, n_trials=args.n_trials,
                                                                        offset=0, amplitude_100=40, fs=1/args.dt_sec, target_snr_db=20,
                                                                        debug_plot=args.debug_plot, add_noise=args.noise[st_idx] > 0)
                            # data[data < 0] = 0

                        elif stimuli_type == 'frequency_pos':
                            data, labels = sweep_frequency_oscillations(frequencies=range_val, n_trials=args.n_trials,
                                                                        offset=0, amplitude_100=4, fs=1 / args.dt_sec,
                                                                        target_snr_db=20,
                                                                        debug_plot=args.debug_plot, add_noise=args.noise[st_idx] > 0
                                                                        , stim_length_sec= args.stim_length_sec)
                            data[data < 0] = 0
                            # plt.figure()
                            # plt.plot(data[0, :, :])
                            # plt.show()

                        elif stimuli_type == 'frequency_neg':
                            data, labels = sweep_frequency_oscillations(frequencies=range_val, n_trials=args.n_trials,
                                                                        offset=0, amplitude_100=40, fs=1 / args.dt_sec,
                                                                        target_snr_db=20,
                                                                        debug_plot=args.debug_plot, add_noise=args.noise[st_idx] > 0)
                            data[data < 0] = 0
                            data = -data
                        elif stimuli_type == 'slope':
                            data, labels = sweep_slopes(slopes=range_val, n_trials=args.n_trials, dt_sec=args.dt_sec, stim_length_sec=args.stim_length_sec,
                                                        last=args.last, first=0, sig=args.noise[st_idx],debug_plot=args.debug_plot)

                        else:
                            raise ValueError('Stimuli type not recognized')

                        torch.save(data, f'experiments/results/{exp_id}/{stimuli}_data.pt')
                        data = data[0, :, :].T
                        concurrency = 1
                        sema = Semaphore(concurrency)
                        all_processes = []
                        for MNclass in MNclasses:
                            # pass
                            calculate_MI_class(stimuli, range_val, MNclass,data, labels,args.dt_sec,name,writer,args)


