import torch
from torch.utils.data import DataLoader
from torchnmf.nmf import NMF
import matplotlib.pyplot as plt
from time import localtime, strftime
from tqdm import trange
from utils import addToNetMetadata, addHeaderToMetadata, set_results_folder, generate_dict
import matplotlib

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
sweep_param_name = ['a', 'A1', 'A2', 'b', 'G', 'k1', 'k2']
sweep_ranges = [[-10, 10], [-100, 100], [-1000, 1000]]

MNclasses = {
    'A2B': {'a':0,'A1':0,'A2': 0},
    'C2J': {'a':5,'A1':0,'A2': 0},
    'K': {'a':30,'A1':0,'A2': 0},
    'L': {'a':30,'A1':10,'A2': -0.6},
    'M2O': {'a':5,'A1':10,'A2': -0.6},
    'P2Q': {'a':5,'A1':5,'A2': -0.3},
    'R': {'a':0,'A1':8,'A2': -0.1},
    'S': {'a':5,'A1':-3,'A2': 0.5},
    'T': {'a':-80,'A1':0,'A2': 0}
}

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


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


class NlinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, H=10):
        super(NlinearRegression, self).__init__()
        self.NLregression = torch.nn.Sequential(
            torch.nn.Linear(inputSize, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, outputSize)
        )

    def forward(self, x):
        out = self.NLregression(x)
        return out


def sim_batch(dataset, device, neuron, varying_element, rank_NMF, model, training={}, list_loss=[], list_mi=[],
              final=False, results_dir=None, net=None):
    # training has optimizer,criterion

    counter = 0
    list_loss_local = []
    list_mi_local = []

    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)

        # x_local = x_local[0, :] * torch.ones_like(x_local)
        # Reset all the layers in the network
        neuron.reset()
        s_out_rec = []
        vmem_neuron = torch.zeros(x_local.shape[0], x_local.shape[1], 1, varying_element.shape[0], x_local.shape[2])
        thr_neuron = torch.zeros_like(vmem_neuron)

        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None] * 1000)  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
            vmem_neuron[:, t] = neuron.state.V  # shape: n_batches x 1 fanout x n_param_values x n_channels
            thr_neuron[:, t] = neuron.state.Thr

        s_out_rec_train = torch.stack(s_out_rec, dim=1)

        # if run_with_fake_input:
        # Overwrite input data with fake one
        s_out_rec_train = torch.zeros_like(s_out_rec_train)
        step_t = 15
        for i in range(params['n_param_values']):
            s_out_rec_train[:, i * step_t:(i + 1) * (step_t), :, i, :] = 1
        #    # print(i*step_t)
        # s_out_rec_train shape: trial x time x fanout x variable x channels

        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))

        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        label = torch.ones([s_out_rec_train.shape[1], varying_element.shape[0]]) * varying_element
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        label = label.T.flatten()[:, None]
        label_min = label.min()
        label_diff = label.max() - label.min()
        label_norm = (label - label_min) / (label_diff)
        H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        selected_neuron_id = 4
        # net = None
        for neuron_id in [selected_neuron_id]:
            V_matrix = s_out_rec_train[:, 0, neuron_id, :]
            if net is None:
                net = NMF(V_matrix.shape, rank=rank_NMF)
            net.fit(V_matrix)
            H[neuron_id] = net.H
            plt.imshow(H[neuron_id].clone().detach(), aspect='auto', interpolation='nearest')
            plt.show()
            if len(training):
                counter += 1
                training['optimizer'].zero_grad()
                outputs = model(H[neuron_id])
                outputs_norm = (outputs - label_min) / (label_diff)

                loss = training['criterion'](outputs, label)
                print('loss', loss)
                # get gradients w.r.t to parameters
                loss.backward()
                list_loss_local.append(loss.clone().detach())
                # update parameters
                training['optimizer'].step()
                acc = ((outputs_norm - label_norm) ** 2).sum() / x_local.shape[0]
                print('acc', acc)
            else:
                predicted = model(H[neuron_id])
                output_vs_label = [predicted.clone().detach(),
                                   label.clone().detach()]
                outputs_norm = (predicted - label_min) / (label_diff)
                acc = ((outputs_norm - label_norm) ** 2).sum() / x_local.shape[0]
                print('acc', acc)

                label_unique = torch.unique(label)
                predicted_int = predicted.type(torch.int)
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

                if final == True:
                    f = plt.figure()
                    plt.imshow(pdf_x1x2)
                    plt.xticks([i for i in range(len(predicted_range))], np.array(predicted_range))
                    plt.yticks([i for i in range(len(label_unique))], np.array(label_unique))
                    plt.xlabel('Predicted')
                    plt.ylabel('Label')
                    plt.title('Prob matrix')
                    plt.colorbar()
                    if save_out:
                        f.savefig(results_dir + 'pdf_joint.pdf', format='pdf')

                    f = plt.figure()
                    plt.imshow(x_local[:, selected_neuron_id, :], aspect='auto')
                    plt.title('XLOCAL')
                    if save_out:
                        f.savefig(results_dir + 'xlocal.pdf', format='pdf')

                    f = plt.figure()
                    plt.imshow(s_out_rec_train[:, 0, selected_neuron_id, :], aspect='auto')
                    # plt.ylim([0, 128*2])
                    plt.title('INPUT NMF')
                    plt.xlabel('Time')
                    plt.ylabel('TrialxVariable')
                    if save_out:
                        f.savefig(results_dir + 'Input_nmf.pdf', format='pdf')

                    # s_out_rec_train shape:  (trial x variable) x fanout x channels x time
                    print('s_out_train shape', s_out_rec_train.shape)
                    f = plt.figure()
                    plt.imshow(H[neuron_id].clone().detach(), aspect='auto', interpolation='nearest')
                    # plt.ylim([0, 128*2])
                    plt.title('H NMF')
                    plt.xlabel('Time')
                    plt.ylabel('TrialxVariable')
                    if save_out:
                        f.savefig(results_dir + 'H.pdf', format='pdf')

                    f = plt.figure()
                    plt.title('W NMF')
                    plt.imshow(net.W.clone().detach(), aspect='auto', interpolation='nearest')
                    plt.xlabel('Time')
                    plt.ylabel('Rank')
                    if save_out:
                        f.savefig(results_dir + 'W_nmf.pdf', format='pdf')

                    if run_with_fake_input:
                        # Store V input:
                        with open(results_dir + 'W.pickle', 'wb') as f:
                            pickle.dump(net.W.clone().detach().numpy(), f)

                    f = plt.figure()
                    plt.imshow(net().clone().detach(), aspect='auto', interpolation='nearest')
                    # plt.ylim([0, 128*2])
                    plt.title('OUT NMF')
                    plt.xlabel('Time')
                    plt.ylabel('TrialxVariable')
                    if save_out:
                        f.savefig(results_dir + 'Out_nmf.pdf', format='pdf')

                    f = plt.figure()
                    plt.imshow(torch.concat(output_vs_label, dim=1), aspect='auto', cmap='seismic',
                               interpolation='nearest')
                    plt.colorbar()
                    plt.title('OUTPUT vs LABEL')
                    # plt.xlabel('Output|Label')
                    plt.ylabel('TrialxVariable')
                    plt.xticks([0, 1], ['Output', 'Label'])
                    if save_out:
                        f.savefig(results_dir + 'Predicted_vs_label.pdf', format='pdf')

                    if run_with_fake_input:
                        # Store Out nmf input:
                        with open(results_dir + 'Out_nmf.pickle', 'wb') as f:
                            pickle.dump(net().clone().detach().numpy(), f)
                        # Store Out classifier:
                        with open(results_dir + 'Out_classifier.pickle', 'wb') as f:
                            pickle.dump(predicted.clone().detach().numpy(), f)
                        # Store Learned W coefficients:
                        with open(results_dir + 'w_classifier.pickle', 'wb') as f:
                            coeff = [p for p in model.parameters()][0][0]
                            coeff = coeff.clone().detach()
                            pickle.dump(coeff[:, None].clone().detach().numpy(), f)
                        # Store joint probability matrix:
                        with open(results_dir + 'pdf_x1x2.pickle', 'wb') as f:
                            pickle.dump(pdf_x1x2.clone().detach().numpy(), f)

                    # plt.show()

                mi = torch.zeros(1)
                for el1_idx, pdf_x1_el in enumerate(pdf_x1):
                    for el2_idx, pdf_x2_el in enumerate(pdf_x2):
                        mi += pdf_x1x2[el1_idx, el2_idx] * torch.log2(
                            (pdf_x1x2[el1_idx, el2_idx] / (pdf_x1_el * pdf_x2_el)) + 1E-10)
                print('mutual information', mi)
                # plt.figure()
                # plt.imshow(pdf_x1x2,aspect='auto')
                # plt.title('PDF')
                # plt.figure()
                # plt.plot(label, predicted_int)
                # plt.show()
                list_mi_local.append(mi)
    if len(training):
        list_loss.append(torch.mean(torch.Tensor(list_loss_local)))
        return list_loss, net
    else:
        list_mi.append(torch.mean(torch.Tensor(list_mi_local)))
        return list_mi, net


def run(dataset, device, neuron, varying_element, rank_NMF, model, training={}, list_loss=[],
                       results_dir=None, net=None, neuron_id=4, epoch=0):


    return list_loss, net


def sim_batch_testing(dataset, device, neuron, varying_element, rank_NMF, model, list_mi=[],
                      final=False, results_dir=None, net=None, neuron_id=4, epoch=0):
    list_mi_local = []
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None] * 1000)  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))
        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        label = torch.ones([s_out_rec_train.shape[1], varying_element.shape[0]]) * varying_element
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        label = label.T.flatten()[:, None]
        label_min = label.min()
        label_diff = label.max() - label.min()
        label_norm = (label - label_min) / (label_diff)
        H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        selected_neuron_id = 4
        # net = None
        V_matrix = s_out_rec_train[:, 0, neuron_id, :]
        net = NMF(V_matrix.shape, rank=rank_NMF)
        net.fit(V_matrix)
        H[neuron_id] = net.H
        predicted = model(H[neuron_id])
        label_unique = torch.unique(label)
        predicted_int = predicted.type(torch.int)
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
        list_mi.append([mi.item(), epoch])
    return list_mi, net


def sim_batch_final(dataset, device, neuron, varying_element, rank_NMF, model, list_mi=[], results_dir=None, net=None,
                    neuron_id=4, epoch=0):
    list_mi_local = []
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None] * 1000)  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))
        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        label = torch.ones([s_out_rec_train.shape[1], varying_element.shape[0]]) * varying_element
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        label = label.T.flatten()[:, None]
        label_min = label.min()
        label_diff = label.max() - label.min()
        label_norm = (label - label_min) / (label_diff)
        H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        V_matrix = s_out_rec_train[:, 0, neuron_id, :]
        # if net is None:
        net = NMF(V_matrix.shape, rank=rank_NMF)
        net.fit(V_matrix)
        H[neuron_id] = net.H
        predicted = model(H[neuron_id])
        output_vs_label = [predicted.clone().detach(),
                           label.clone().detach()]
        outputs_norm = (predicted - label_min) / (label_diff)
        acc = ((outputs_norm - label_norm) ** 2).sum() / x_local.shape[0]
        label_unique = torch.unique(label)
        predicted_int = predicted.type(torch.int)
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
        f = plt.figure()
        plt.imshow(pdf_x1x2.clone().detach().cpu())
        plt.xticks([i for i in range(len(predicted_range))], np.array(predicted_range.cpu()))
        plt.yticks([i for i in range(len(label_unique))], np.array(label_unique.cpu()))
        plt.xlabel('Predicted')
        plt.ylabel('Label')
        plt.title('Prob matrix')
        plt.colorbar()
        if save_out:
            f.savefig(results_dir + 'pdf_joint.pdf', format='pdf')

        f = plt.figure()
        plt.imshow(x_local[:, neuron_id, :].clone().detach().cpu(), aspect='auto')
        plt.title('XLOCAL')
        if save_out:
            f.savefig(results_dir + 'xlocal.pdf', format='pdf')

        f = plt.figure()
        plt.imshow(s_out_rec_train[:, 0, neuron_id, :].clone().detach().cpu(), aspect='auto')
        # plt.ylim([0, 128*2])
        plt.title('INPUT NMF')
        plt.xlabel('Time')
        plt.ylabel('TrialxVariable')
        if save_out:
            f.savefig(results_dir + 'Input_nmf.pdf', format='pdf')

        # s_out_rec_train shape:  (trial x variable) x fanout x channels x time
        # print('s_out_train shape', s_out_rec_train.shape)
        f = plt.figure()
        plt.imshow(H[neuron_id].clone().detach().cpu(), aspect='auto', interpolation='nearest')
        # plt.ylim([0, 128*2])
        plt.title('H NMF')
        plt.xlabel('Time')
        plt.ylabel('TrialxVariable')
        if save_out:
            f.savefig(results_dir + 'H.pdf', format='pdf')

        f = plt.figure()
        plt.title('W NMF')
        plt.imshow(net.W.clone().detach().cpu(), aspect='auto', interpolation='nearest')
        plt.xlabel('Time')
        plt.ylabel('Rank')
        if save_out:
            f.savefig(results_dir + 'W_nmf.pdf', format='pdf')

        if run_with_fake_input:
            # Store V input:
            with open(results_dir + 'W.pickle', 'wb') as f:
                pickle.dump(net.W.clone().detach().cpu()(), f)

        f = plt.figure()
        plt.imshow(net().clone().detach().cpu(), aspect='auto', interpolation='nearest')
        # plt.ylim([0, 128*2])
        plt.title('OUT NMF')
        plt.xlabel('Time')
        plt.ylabel('TrialxVariable')
        if save_out:
            f.savefig(results_dir + 'Out_nmf.pdf', format='pdf')

        f = plt.figure()
        plt.imshow(torch.concat(output_vs_label, dim=1).clone().detach().cpu(), aspect='auto', cmap='seismic',
                   interpolation='nearest')
        plt.colorbar()
        plt.title('OUTPUT vs LABEL')
        # plt.xlabel('Output|Label')
        plt.ylabel('TrialxVariable')
        plt.xticks([0, 1], ['Output', 'Label'])
        if save_out:
            f.savefig(results_dir + 'Predicted_vs_label.pdf', format='pdf')

        if run_with_fake_input:
            # Store Out nmf input:
            with open(results_dir + 'Out_nmf.pickle', 'wb') as f:
                pickle.dump(net().clone().detach().cpu(), f)
            # Store Out classifier:
            with open(results_dir + 'Out_classifier.pickle', 'wb') as f:
                pickle.dump(predicted.clone().detach().cpu(), f)
            # Store Learned W coefficients:
            with open(results_dir + 'w_classifier.pickle', 'wb') as f:
                coeff = [p for p in model.parameters()][0][0]
                coeff = coeff.clone().detach()
                pickle.dump(coeff[:, None].clone().detach().cpu(), f)
            # Store joint probability matrix:
            with open(results_dir + 'pdf_x1x2.pickle', 'wb') as f:
                pickle.dump(pdf_x1x2.clone().detach().cpu(), f)
        mi = torch.zeros(1)
        for el1_idx, pdf_x1_el in enumerate(pdf_x1):
            for el2_idx, pdf_x2_el in enumerate(pdf_x2):
                mi += pdf_x1x2[el1_idx, el2_idx] * torch.log2(
                    (pdf_x1x2[el1_idx, el2_idx] / (pdf_x1_el * pdf_x2_el)) + 1E-10)
                # print('mutual information', mi)
                # plt.figure()
                # plt.imshow(pdf_x1x2,aspect='auto')
                # plt.title('PDF')
                # plt.figure()
                # plt.plot(label, predicted_int)
                # plt.show()
                list_mi.append([mi.item(), epoch])
    # list_mi.append(torch.mean(torch.Tensor(list_mi_local)))
    return list_mi, net


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
                                 target_snr_db=20, debug_plot=True, add_noise=True):
    """
    Return amplitude of input current across time, with as many input signals as the dimension of
    the input amplitudes.

    dt_sec:
    stim_length_sec:
    amplitudes
    """

    n_neurons = len(frequencies) * n_trials
    stim = []
    area = amplitude_100/np.pi/100/2
    list_mean_frequency = []  # list with mean current value (same dimension as the last dimension of input_current)
    for f in frequencies:
        t = np.arange(fs) / fs
        n_time_bins = len(t)
        amplitude = area*np.pi*f*2
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

    return input_current, list_mean_frequency

def train(dl_train, neuron, params, device,rank_NMF,optimizer,model,criterion,writer,epoch):
    list_epoch_loss = []
    for x_local, y_local in dl_train:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        params['nb_channels'] = x_local.shape[0]
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(
                x_local[:, t, None, None, None])  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))

        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        V_matrix = s_out_rec_train[:, 0, 0, :]
        net = NMF(V_matrix.shape, rank=rank_NMF)
        net.fit(V_matrix)
        H[0] = net.H
        optimizer.zero_grad()
        outputs = model(net.H)
        # outputs_norm = (outputs - label_min) / (label_diff)
        loss = criterion(outputs[:,0], y_local)
        # get gradients w.r.t to parameters
        loss.backward()
        list_epoch_loss.append(loss.item())
        # update parameters
        optimizer.step()
    writer.add_scalar('MSE/train', torch.mean(torch.tensor(list_epoch_loss)).cpu().numpy(), epoch)
    return torch.mean(torch.tensor(list_epoch_loss)).clone().cpu().numpy()
        # print('Epoch: {}. MSE train: {}. '.format(e, loss.item()))

def eval(dl_test, neuron, params, device,rank_NMF,model,criterion,writer,epoch):

    list_epoch_loss_test = []
    list_epoch_MI = []
    for x_local, y_local in dl_test:
        with torch.no_grad():
            x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
            params['nb_channels'] = x_local.shape[0]
            neuron.reset()
            s_out_rec = []
            for t in range(x_local.shape[1]):
                out = neuron(
                    x_local[:, t, None, None, None])  # shape: n_batches x 1 fanout x n_param_values x n_channels
                s_out_rec.append(out)
            s_out_rec_train = torch.stack(s_out_rec, dim=1)
            s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))

            ### s_out_rec_train shape:  trial x variable x fanout x channels x time
            s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
            H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
            V_matrix = s_out_rec_train[:, 0, 0, :]
        net = NMF(V_matrix.shape, rank=rank_NMF)
        net.fit(V_matrix)
        H[0] = net.H
        with torch.no_grad():

            predicted = model(net.H)
            loss_test = criterion(predicted[:,0], y_local)
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
    fig3, axis3 = plt.subplots()
    pcm = axis1.imshow([predicted.cpu().numpy(), label[:, None].cpu().numpy()], aspect='auto')
    # axis1.colorbar()
    fig1.colorbar(pcm, ax=axis1)
    pcm2 = axis2.imshow(pdf_x1x2.cpu().numpy(), aspect='auto')
    fig2.colorbar(pcm2, ax=axis2)
    eee = torch.argsort(y_local)
    aaa = torch.where(s_out_rec_train[eee, 0, 0, :])
    axis3.scatter(aaa[1].cpu().numpy(),aaa[0].cpu().numpy())
    writer.add_figure(figure=fig1, global_step=epoch, tag='test_resp')
    writer.add_figure(figure=fig2, global_step=epoch, tag='test_mi')
    writer.add_figure(figure=fig3, global_step=epoch, tag='test_V')
    writer.add_scalar('MI', torch.mean(torch.tensor(list_epoch_MI)).cpu().numpy(), epoch)
    # writer.add_scalar('MSE/train', torch.mean(torch.tensor(list_epoch_loss)).cpu().numpy(), e)
    writer.add_scalar('MSE/test', torch.mean(torch.tensor(list_epoch_loss_test)).cpu().numpy(), epoch)
    return torch.mean(torch.tensor(list_epoch_MI)).cpu().numpy(), torch.mean(torch.tensor(list_epoch_loss_test)).cpu().numpy()
def MI_neuron_params(neuron_param_values, name_param_sweeped, extremes_sweep, MNclass,data,labels,dt_sec):
    # Set results folder:

    iscuda = torch.cuda.is_available()
    # # Forcing CPU
    # iscuda = False
    # print('NOTE: Forced CPU')

    if run_with_fake_input:
        results_dir = 'results/controlled_stim/'
        if not (os.path.isdir(results_dir)):
            os.mkdir(results_dir)
    else:
        results_dir = set_results_folder([exp_id,MNclass, name_param_sweeped, str(extremes_sweep)])
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
    params['nb_channels'] = 1
    params['labels'] = labels

    params['nb_channels'] = 1
    params['data_steps'] = dt_sec

    # Network parameters
    # Learning parameters
    nb_epochs = int(params['nb_epochs'])
    ###########################################
    ##                Network                ##
    ###########################################

    n_param_values = params['n_param_values']

    tensor_params = dict.fromkeys(neuron_param_values.keys(), None)
    for key in tensor_params.keys():
        tensor_params[key] = torch.Tensor(neuron_param_values[key]).to(device)

    # varying_element = tensor_params[name_param_sweeped]

    # a = torch.Tensor(neuron_param_values['a']).to(device)
    # # nn.init.normal_(
    # #     a, mean=MNparams_dict['A2B'][0], std=fwd_weight_scale / np.sqrt(nb_inputs))
    #
    # A1 = torch.Tensor(neuron_param_values['A1']).to(device)
    #
    # A2 = torch.Tensor(neuron_param_values['A2']).to(device)

    fanout = 1  # number of output neurons from the linear expansion
    # TODO: Change input parameters list
    neuron = models.MN_neuron_IT(params['nb_channels'], fanout, 1,
                                 tensor_params['a'],
                                 tensor_params['A1'],
                                 tensor_params['A2'],
                                 tensor_params['b'],
                                 tensor_params['G'],
                                 tensor_params['k1'],
                                 tensor_params['k2'],
                                 train=False,dt=dt_sec)

    batch_size = int(params['batch_size'])

    inputDim = 1  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    learningRate = params['learningRate']  # 10#0.01

    neuron_id = int(params['neuron_id'])

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
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          generator=torch.Generator(device=device))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=num_workers,
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

    list_loss = []
    list_mi = []
    t_start = time.time()
    H_initial = None
    # net = NMF((), rank=rank_NMF, H=H_initial)
    net = None
    writer = SummaryWriter(comment="Stim" + str(name_param_sweeped) + "_MN_class"+str(MNclass))

    pbar = trange(nb_epochs)
    for e in pbar:
        list_epoch_loss = []
        list_epoch_MI = []
        list_epoch_loss_test = []
        loss_train = train(epoch=e, model=model, optimizer=optimizer, criterion=criterion, dl_train=dl_train, device=device,rank_NMF=rank_NMF, writer=writer,neuron=neuron,params=params)
        # list_loss.append(torch.mean(torch.tensor(list_epoch_loss)).cpu().numpy())

                # get gradients w.r.t to parameters
                # update parameters
        mi_test, mse_test = eval(epoch=e, model=model, criterion=criterion, dl_test=dl_test, device=device,rank_NMF=rank_NMF, writer=writer,neuron=neuron,params=params)
        # list_mi.append(torch.mean(torch.tensor(list_epoch_mi)).cpu().numpy())
        # print('plotting')
        # fig1,axis1 = plt.subplots()
        # plt.figure()
        # axis1.plot(list_loss,color='b')
        # axis2 = axis1.twinx()
        # axis2.plot(list_mi, color='r')
        # plt.show()
    torch.save(loss_train, results_dir + 'Loss' + MNclass + '.pt')
    torch.save(mi_test, results_dir + 'MI' + MNclass + '.pt')
    torch.save(mse_test, results_dir + 'MSE' + MNclass + '.pt')
if __name__ == "__main__":
    print('Current path',Current_PATH)
    for MNclass in MNclasses:
            print('-------------------------------------------')
            print('Class {}'.format(MNclass))
            # Generate dictionary with parameter values:
            upsample_fac = 5
            stim_length_sec = 0.1
            dt_sec = 0.001
            n_time_bins = int(np.floor(stim_length_sec / dt_sec))
            n_trials = 100
            amplitudes = np.arange(10)
            data, labels = sweep_steps(amplitudes=amplitudes, n_trials=n_trials, dt_sec=dt_sec,
                                       stim_length_sec=stim_length_sec, sig=0.01, debug_plot=True)
            data = data[0, :, :].T
            dict_keys = generate_dict('a', [0],force_param_dict=MNclasses[MNclass])

            # Run mutual information analysis
            # MI_neuron_params(dict_keys, 'ampli', [dict_keys['a'],dict_keys['a']], MNclass,data,labels,dt_sec=dt_sec)
            freqs = np.arange(100,400,100)
            data, labels = sweep_frequency_oscillations(frequencies=freqs, n_trials=n_trials,
                                                                    offset=1, amplitude_100=1, fs=1/dt_sec, target_snr_db=20,
                                                                    debug_plot=False, add_noise=False)

            data = data[0, :, :].T
            plt.figure()
            for i in range(300):
                fft = np.abs(np.fft.fft(data[i, :]))
                plt.plot(fft[:int(len(fft) / 2)])
            plt.show()
            dict_keys = generate_dict('a', [0], force_param_dict=MNclasses[MNclass])

            # Run mutual information analysis
            MI_neuron_params(dict_keys, 'freqs', [dict_keys['a'], dict_keys['a']], MNclass, data, labels, dt_sec=dt_sec)
