import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchnmf.nmf import NMF
from torchnmf.metrics import kl_div
import matplotlib.pyplot as plt
from time import localtime, strftime
from tqdm import trange
from utils import addToNetMetadata, addHeaderToMetadata, set_results_folder, generate_dict
import matplotlib
matplotlib.pyplot.ioff() # turn off interactive mode
import numpy as np
import os
import models
from datasets import load_analog_data
from parameters.MN_params import MNparams_dict, INIT_MODE
from models import Encoder, LIF_neuron, MN_neuron_IT
from auxiliary import compute_classification_accuracy, plot_spikes
import time
from torch.autograd import Variable
import pickle
import scipy.spatial as sp
matplotlib.use('Agg')
# Set seed:
torch.manual_seed(0)

# ---------------------------- Input -----------------------------------
save_out = True # Flag to save figures:
sweep_param_name = ['a', 'A1', 'A2']

run_with_fake_input = False
# ---------------------------- Parameters -----------------------------------
threshold = "enc"
run = "_3"

file_dir_params = '../parameters/'
param_filename = 'parameters_th' + str(threshold)
file_name_parameters = file_dir_params + param_filename + '.txt'
params = {}
with open(file_name_parameters) as file:
    for line in file:
        (key, value) = line.split()
        if key == 'time_bin_size' or key == 'nb_input_copies' or key=='n_param_values' or key=='min_range' or key=='max_range':
            params[key] = int(value)
        else:
            params[key] = np.double(value)

variable_range = np.linspace(params['min_range'], params['max_range'], params['n_param_values'])


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
                            torch.nn.Linear(H,outputSize)
        )
    def forward(self, x):
        out = self.NLregression(x)
        return out

def sim_batch(dataset, device, neuron, varying_element, rank_NMF, model, training ={}, list_loss=[], list_mi=[],
              final = False, results_dir=None, net = None):
    #training has optimizer,criterion
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
            out = neuron(x_local[:, t, None, None]*1000)  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
            vmem_neuron[:,t] = neuron.state.V # shape: n_batches x 1 fanout x n_param_values x n_channels
            thr_neuron[:,t] = neuron.state.Thr

        s_out_rec_train = torch.stack(s_out_rec, dim=1)

        # if run_with_fake_input:
            # Overwrite input data with fake one
        s_out_rec_train = torch.zeros_like(s_out_rec_train)
        step_t = 15
        for i in range(params['n_param_values']):
            s_out_rec_train[:,i*step_t:(i+1)*(step_t),:,i,:] = 1
        #    # print(i*step_t)
            # s_out_rec_train shape: trial x time x fanout x variable x channels

        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))

        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        label = torch.ones([s_out_rec_train.shape[1], varying_element.shape[0]]) * varying_element
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        label = label.T.flatten()[:, None]
        label_min = label.min()
        label_diff = label.max()-label.min()
        label_norm = (label - label_min)/(label_diff)
        H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        selected_neuron_id = 4
        # net = None
        for neuron_id in [selected_neuron_id]:
            V_matrix = s_out_rec_train[:, 0, neuron_id, :]
            if net is None:
                net = NMF(V_matrix.shape, rank=rank_NMF)
            net.fit(V_matrix)
            H[neuron_id] = net.H
            plt.imshow(H[neuron_id].clone().detach(),aspect = 'auto',interpolation='nearest')
            plt.show()
            if len(training):
                counter += 1
                training['optimizer'].zero_grad()
                outputs = model(H[neuron_id])
                outputs_norm = (outputs - label_min) / (label_diff)

                loss = training['criterion'](outputs, label)
                print('loss',loss)
                # get gradients w.r.t to parameters
                loss.backward()
                list_loss_local.append(loss.clone().detach())
                # update parameters
                training['optimizer'].step()
                acc = ((outputs_norm - label_norm) ** 2).sum()/x_local.shape[0]
                print('acc', acc)
            else:
                predicted = model(H[neuron_id])
                output_vs_label = [predicted.clone().detach(),
                                   label.clone().detach()]
                outputs_norm = (predicted - label_min) / (label_diff)
                acc = ((outputs_norm - label_norm) ** 2).sum()/x_local.shape[0]
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
                pdf_x1 = torch.sum(pdf_x1x2, dim=1)/num_occ # to check
                pdf_x2 = torch.sum(pdf_x1x2, dim=0)/num_occ
                pdf_x1x2 = pdf_x1x2 / num_occ

                if final == True:
                    f = plt.figure()
                    plt.imshow(pdf_x1x2)
                    plt.xticks([i for i in range(len(predicted_range))],np.array(predicted_range))
                    plt.yticks([i for i in range(len(label_unique))],np.array(label_unique))
                    plt.xlabel('Predicted')
                    plt.ylabel('Label')
                    plt.title('Prob matrix')
                    plt.colorbar()
                    if save_out:
                        f.savefig(results_dir+'pdf_joint.pdf', format='pdf')

                    f = plt.figure()
                    plt.imshow(x_local[:, selected_neuron_id, :], aspect='auto')
                    plt.title('XLOCAL')
                    if save_out:
                        f.savefig(results_dir+'xlocal.pdf', format='pdf')

                    f = plt.figure()
                    plt.imshow(s_out_rec_train[:, 0, selected_neuron_id, :], aspect='auto')
                    # plt.ylim([0, 128*2])
                    plt.title('INPUT NMF')
                    plt.xlabel('Time')
                    plt.ylabel('TrialxVariable')
                    if save_out:
                        f.savefig(results_dir+'Input_nmf.pdf', format='pdf')

                    # s_out_rec_train shape:  (trial x variable) x fanout x channels x time
                    print('s_out_train shape', s_out_rec_train.shape)
                    f = plt.figure()
                    plt.imshow(H[neuron_id].clone().detach(), aspect='auto', interpolation='nearest')
                    # plt.ylim([0, 128*2])
                    plt.title('H NMF')
                    plt.xlabel('Time')
                    plt.ylabel('TrialxVariable')
                    if save_out:
                        f.savefig(results_dir+'H.pdf', format='pdf')

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

                    f=plt.figure()
                    plt.imshow(net().clone().detach(), aspect='auto', interpolation='nearest')
                    # plt.ylim([0, 128*2])
                    plt.title('OUT NMF')
                    plt.xlabel('Time')
                    plt.ylabel('TrialxVariable')
                    if save_out:
                        f.savefig(results_dir+'Out_nmf.pdf', format='pdf')

                    f = plt.figure()
                    plt.imshow(torch.concat(output_vs_label, dim=1), aspect='auto', cmap='seismic',
                               interpolation='nearest')
                    plt.colorbar()
                    plt.title('OUTPUT vs LABEL')
                    # plt.xlabel('Output|Label')
                    plt.ylabel('TrialxVariable')
                    plt.xticks([0, 1], ['Output', 'Label'])
                    if save_out:
                        f.savefig(results_dir+'Predicted_vs_label.pdf', format='pdf')

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

def sim_batch_training(dataset, device, neuron, varying_element, rank_NMF, model, training ={}, list_loss=[], results_dir=None, net = None, neuron_id = 4, epoch = 0):
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None]*1000)  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))

        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        label = torch.ones([s_out_rec_train.shape[1], varying_element.shape[0]]) * varying_element
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        label = label.T.flatten()[:, None]
        label_min = label.min()
        label_diff = label.max()-label.min()
        label_norm = (label - label_min)/(label_diff)
        H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        V_matrix = s_out_rec_train[:, 0, neuron_id, :]
        net = NMF(V_matrix.shape, rank=rank_NMF)
        net.fit(V_matrix)
        H[neuron_id] = net.H
        training['optimizer'].zero_grad()
        outputs = model(H[neuron_id])
        outputs_norm = (outputs - label_min) / (label_diff)
        loss = training['criterion'](outputs, label)
        # get gradients w.r.t to parameters
        loss.backward()
        list_loss.append([loss.item(),epoch])
        # update parameters
        training['optimizer'].step()
    return list_loss, net

def sim_batch_testing(dataset, device, neuron, varying_element, rank_NMF, model, list_mi=[],
              final = False, results_dir=None, net = None, neuron_id = 4, epoch = 0):
    list_mi_local = []
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None]*1000)  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))
        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        label = torch.ones([s_out_rec_train.shape[1], varying_element.shape[0]]) * varying_element
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        label = label.T.flatten()[:, None]
        label_min = label.min()
        label_diff = label.max()-label.min()
        label_norm = (label - label_min)/(label_diff)
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
        pdf_x1 = torch.sum(pdf_x1x2, dim=1)/num_occ # to check
        pdf_x2 = torch.sum(pdf_x1x2, dim=0)/num_occ
        pdf_x1x2 = pdf_x1x2 / num_occ
        mi = torch.zeros(1)
        for el1_idx, pdf_x1_el in enumerate(pdf_x1):
            for el2_idx, pdf_x2_el in enumerate(pdf_x2):
                mi += pdf_x1x2[el1_idx, el2_idx] * torch.log2(
                    (pdf_x1x2[el1_idx, el2_idx] / (pdf_x1_el * pdf_x2_el)) + 1E-10)
        list_mi.append([mi.item(),epoch])
    return list_mi, net

def sim_batch_final(dataset, device, neuron, varying_element, rank_NMF, model, list_mi=[],results_dir=None, net = None,
                    neuron_id = 4, epoch = 0):
    list_mi_local = []
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None]*1000)  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))
        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        label = torch.ones([s_out_rec_train.shape[1], varying_element.shape[0]]) * varying_element
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        label = label.T.flatten()[:, None]
        label_min = label.min()
        label_diff = label.max()-label.min()
        label_norm = (label - label_min)/(label_diff)
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
        acc = ((outputs_norm - label_norm) ** 2).sum()/x_local.shape[0]
        label_unique = torch.unique(label)
        predicted_int = predicted.type(torch.int)
        predicted_range = torch.unique(predicted_int)



        pdf_x1x2 = torch.zeros([len(label_unique), len(predicted_range)])
        for trial_idx in range(len(predicted_int)):

            lab_pos = torch.where(label_unique == label[trial_idx])[0]
            pred_pos = torch.where(predicted_range == predicted_int[trial_idx])[0]
            pdf_x1x2[lab_pos, pred_pos] += 1

        num_occ = torch.sum(pdf_x1x2)
        pdf_x1 = torch.sum(pdf_x1x2, dim=1)/num_occ # to check
        pdf_x2 = torch.sum(pdf_x1x2, dim=0)/num_occ
        pdf_x1x2 = pdf_x1x2 / num_occ
        f = plt.figure()
        plt.imshow(pdf_x1x2.clone().detach().cpu())
        plt.xticks([i for i in range(len(predicted_range))],np.array(predicted_range.cpu()))
        plt.yticks([i for i in range(len(label_unique))],np.array(label_unique.cpu()))
        plt.xlabel('Predicted')
        plt.ylabel('Label')
        plt.title('Prob matrix')
        plt.colorbar()
        if save_out:
            f.savefig(results_dir+'pdf_joint.pdf', format='pdf')

        f = plt.figure()
        plt.imshow(x_local[:, neuron_id, :].clone().detach().cpu(), aspect='auto')
        plt.title('XLOCAL')
        if save_out:
            f.savefig(results_dir+'xlocal.pdf', format='pdf')

        f = plt.figure()
        plt.imshow(s_out_rec_train[:, 0, neuron_id, :].clone().detach().cpu(), aspect='auto')
        # plt.ylim([0, 128*2])
        plt.title('INPUT NMF')
        plt.xlabel('Time')
        plt.ylabel('TrialxVariable')
        if save_out:
            f.savefig(results_dir+'Input_nmf.pdf', format='pdf')

        # s_out_rec_train shape:  (trial x variable) x fanout x channels x time
        # print('s_out_train shape', s_out_rec_train.shape)
        f = plt.figure()
        plt.imshow(H[neuron_id].clone().detach().cpu(), aspect='auto', interpolation='nearest')
        # plt.ylim([0, 128*2])
        plt.title('H NMF')
        plt.xlabel('Time')
        plt.ylabel('TrialxVariable')
        if save_out:
            f.savefig(results_dir+'H.pdf', format='pdf')

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

        f=plt.figure()
        plt.imshow(net().clone().detach().cpu(), aspect='auto', interpolation='nearest')
        # plt.ylim([0, 128*2])
        plt.title('OUT NMF')
        plt.xlabel('Time')
        plt.ylabel('TrialxVariable')
        if save_out:
            f.savefig(results_dir+'Out_nmf.pdf', format='pdf')

        f = plt.figure()
        plt.imshow(torch.concat(output_vs_label, dim=1).clone().detach().cpu(), aspect='auto', cmap='seismic',
                   interpolation='nearest')
        plt.colorbar()
        plt.title('OUTPUT vs LABEL')
        # plt.xlabel('Output|Label')
        plt.ylabel('TrialxVariable')
        plt.xticks([0, 1], ['Output', 'Label'])
        if save_out:
            f.savefig(results_dir+'Predicted_vs_label.pdf', format='pdf')

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
                list_mi.append([mi.item(),epoch])
    # list_mi.append(torch.mean(torch.Tensor(list_mi_local)))
    return list_mi, net

def MI_neuron_params(neuron_param_values, name_param_sweeped):

    # Set results folder:

    iscuda = torch.cuda.is_available()
    # #Forcing CPU
    # iscuda = False
    # print('NOTE: Forced CPU')
    if run_with_fake_input:
        results_dir = '../results/controlled_stim/'
        if not (os.path.isdir(results_dir)):
            os.mkdir(results_dir)
    else:
        results_dir = set_results_folder(name_param_sweeped, exp_id)
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
    upsample_fac = 5
    file_name = "../data/data_braille_letters_digits.pkl"
    ds_train, ds_test, labels, nb_channels, data_steps = load_analog_data(file_name, upsample_fac,
                                                                          specify_letters=['A'])
    params['nb_channels'] = nb_channels
    params['labels'] = labels
    params['data_steps'] = data_steps

    # Network parameters
    nb_input_copies = params['nb_input_copies']
    nb_inputs = params['nb_channels'] * nb_input_copies
    nb_hidden = 450
    nb_outputs = len(torch.unique(params['labels']))

    # Learning parameters
    nb_steps = params['data_steps']
    nb_epochs = int(params['nb_epochs'])

    # Neuron parameters
    tau_mem = params['tau_mem']  # ms
    tau_syn = tau_mem / params['tau_ratio']
    alpha = float(np.exp(-params['data_steps'] * 0.001 / tau_syn))
    beta = float(np.exp(-params['data_steps'] * 0.001 / tau_mem))

    encoder_weight_scale = 1.0
    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = params['weight_scale_factor'] * fwd_weight_scale

    ###########################################
    ##                Network                ##
    ###########################################

    # a = torch.empty((nb_inputs,))
    n_param_values = params['n_param_values']
    #a = torch.Tensor(np.linspace(-10, 10, n_param_values)).to(device)

    tensor_params = dict.fromkeys(neuron_param_values.keys(), None)
    for key in tensor_params.keys():
        tensor_params[key] = torch.Tensor(neuron_param_values[key]).to(device)

    varying_element = tensor_params[name_param_sweeped]

    # a = torch.Tensor(neuron_param_values['a']).to(device)
    # # nn.init.normal_(
    # #     a, mean=MNparams_dict['A2B'][0], std=fwd_weight_scale / np.sqrt(nb_inputs))
    #
    # A1 = torch.Tensor(neuron_param_values['A1']).to(device)
    #
    # A2 = torch.Tensor(neuron_param_values['A2']).to(device)

    fanout = 1  # number of output neurons from the linear expansion
    #TODO: Change input parameters list
    neuron = models.MN_neuron_IT(params['nb_channels'], fanout, n_param_values,
                                 tensor_params['a'],
                                 tensor_params['A1'],
                                 tensor_params['A2'],
                                 train=False)

    batch_size = int(params['batch_size'])

    inputDim = 1  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    learningRate = params['learningRate'] #10#0.01

    neuron_id = int(params['neuron_id'])

    rank_NMF = int(params['rank_NMF'])
    range_weight_init = 10
    model = NlinearRegression(rank_NMF, 1)
    #model.linear.weight = torch.nn.Parameter(model.linear.weight*range_weight_init)

    coeff = [p for p in model.parameters()][0][0]
    coeff = coeff.clone().detach()

    #TODO: Check this
    if iscuda:
        pin_memory=True
        num_workers = 0
    else:
        pin_memory=False
        num_workers = 32

    # The log softmax function across output units
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory, generator=torch.Generator(device=device))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory, generator=torch.Generator(device=device))
    # pbar = trange(nb_epochs)
    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    optimizer = torch.optim.Adamax(model.parameters(), lr=learningRate)
    params['optimizer'] = optimizer.__class__.__name__
    print('Training classifier with {} optimizer'.format(params['optimizer']))

    # Store parameters to metadata file:
    for key in params.keys():
        addToNetMetadata(metadatafilename, key, params[key])

    header = 'Neuron params'
    for key in neuron_param_values.keys():
        addToNetMetadata(metadatafilename, key, neuron_param_values[key], header=header)
        header=''

    list_loss = []
    list_mi = []
    t_start = time.time()
    H_initial = None
    # net = NMF((), rank=rank_NMF, H=H_initial)
    net = None
    pbar = trange(nb_epochs)
    for e in pbar:
        local_loss = []
        H_train = []
        H_test = []
        accs = []  # accs: mean training accuracies for each batch
        # print('Epoch', e)
        list_loss,net = sim_batch_training(dl_train, device, neuron, varying_element, rank_NMF, model,
                              {'optimizer':optimizer, 'criterion':criterion}, list_loss=list_loss,
                                           results_dir=results_dir,net = net,neuron_id=neuron_id, epoch = e)
        list_mi,net = sim_batch_testing(dl_test, device, neuron, varying_element, rank_NMF, model, list_mi=list_mi,
                                        results_dir=results_dir,net=net,neuron_id=neuron_id, epoch = e)
        pbar.set_postfix_str("Mutual Information: " + str(np.round(list_mi[e][0],2)) + ' bits. Loss: ' + str(np.round(list_loss[e][0], 2)))
    train_duration = time.time() - t_start
    addToNetMetadata(metadatafilename, 'sim duration (sec)', train_duration)

    list_mi,net = sim_batch_final(dl_test, device, neuron, varying_element, rank_NMF, model, list_mi=list_mi,
                              results_dir=results_dir,neuron_id=neuron_id, epoch = e)

    # Store data:
    # Loss:
    list_loss = np.array(list_loss)
    list_mi = np.array(list_mi)
    with open(results_dir+'Loss.pickle', 'wb') as f:
        pickle.dump(list_loss, f)
    # Mi:
    with open(results_dir+'MI.pickle', 'wb') as f:
        pickle.dump(list_mi, f)

    fig = plt.figure()
    plt.plot(list_loss)
    plt.xlabel('Epochs x Trials')
    plt.ylabel('Loss')
    if save_out:
        fig.savefig(results_dir+'Loss.pdf', format='pdf')

    fig = plt.figure()
    plt.plot(list_mi[:,0])
    plt.xlabel('Epochs x Trials')
    plt.ylabel('MI')
    if save_out:
        fig.savefig(results_dir+'MI.pdf', format='pdf')


if __name__ == "__main__":

    if run_with_fake_input:
        name_param = 'a'
        # Generate dictionary with parameter values:
        dict_keys = generate_dict(name_param, variable_range)

        # Run mutual information analysis
        MI_neuron_params(dict_keys, name_param)

    else:
        for name_param in sweep_param_name:

            # Generate dictionary with parameter values:
            dict_keys = generate_dict(name_param, variable_range)

            # Run mutual information analysis
            MI_neuron_params(dict_keys, name_param)
