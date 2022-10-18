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

# Set seed:
torch.manual_seed(0)

# ---------------------------- Input -----------------------------------
save_out = True # Flag to save figures:
sweep_param_name = ['a', 'A1', 'A2']

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
        if key == 'time_bin_size' or key == 'nb_input_copies' or key=='n_param_values':
            params[key] = int(value)
        else:
            params[key] = np.double(value)

variable_range = np.linspace(-10, 10, params['n_param_values'])


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
              final = False, results_dir=None):
    #training has optimizer,criterion
    counter = 0

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

        # s_out_rec_train = torch.zeros_like(s_out_rec_train)
        # step_t = 15
        # for i in range(10):
        #     s_out_rec_train[:,i*step_t:(i+1)*(step_t),:,i,:] = 1
            #print(i*step_t)
        ## s_out_rec_train shape: trial x time x fanout x variable x channels

        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))

        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        label = torch.ones([s_out_rec_train.shape[1], varying_element.shape[0]]) * varying_element
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        label = label.T.flatten()[:, None]
        label_min = label.min()
        label_diff = label.max()-label.min()
        label_norm = (label - label_min)/(label_diff)
        # print('label_norm min max',label_norm.min(),label_norm.max())

        # plt.show()

        H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        selected_neuron_id = 4
        for neuron_id in [selected_neuron_id]:
            V_matrix = s_out_rec_train[:, 0, neuron_id, :]

            net = NMF(V_matrix.shape, rank=rank_NMF)
            net.fit(V_matrix.to_sparse())
            #batch_size = 20
            # H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])

            H[neuron_id] = net.H
            # for i in range(rank_NMF):
            #     H[neuron_id, i * batch_size:(i + 1) * (batch_size),i] = 1

            import scipy.spatial as sp
            cdist = 1 - sp.distance.cdist(net().clone().detach().cpu(), V_matrix.cpu().clone().detach().cpu(), 'cosine')
            diagonal_cdist = cdist.diagonal()
            diag_cdist_nonan = diagonal_cdist[np.isnan(diagonal_cdist) == False]

            # print('gigi diag',np.mean(diag_cdist_nonan))
            # plt.imshow(cdist,aspect = 'auto')
            # plt.colorbar()
            # plt.show()
            # plt.show()

            if len(training):
                # print('training total spikes for neuron', neuron_id, ':', V_matrix.sum())
                counter += 1
                training['optimizer'].zero_grad()
                outputs = model(H[neuron_id])
                outputs_norm = (outputs - label_min) / (label_diff)
                output_vs_label = [outputs.clone().detach(),
                                   label.clone().detach()]
                coeff = [p for p in model.parameters()][0][0]
                coeff = coeff.clone().detach()
                #plt.plot(coeff)
                if counter == -1:

                    idx_param_value = 0
                    idx_fanout = 0
                    for trial in range(3):#range(vmem_neuron.shape[0]):

                        p = plt.plot(thr_neuron[trial,:,idx_fanout,idx_param_value,selected_neuron_id])
                        idx_out_spikes_in_trial = s_out_rec_train[idx_param_value*x_local.shape[0]+trial,idx_fanout, neuron_id,:]==1
                        v = vmem_neuron[trial,:,idx_fanout,idx_param_value,selected_neuron_id]

                        plt.vlines(np.arange(x_local.shape[1])[idx_out_spikes_in_trial], float(v.min()), float(v.max()), color=p[0].get_color(), linestyle='dashed')

                        plt.plot(v, label=str(len(np.where(idx_out_spikes_in_trial==True)[0])), linestyle='dotted', color=p[0].get_color())

                        plt.legend()

                    plt.title('V neuron across trials')
                    plt.xlabel('Time')
                    plt.ylabel('vmem')

                    plt.figure()
                    plt.imshow(x_local[:,4,:],aspect = 'auto')
                    plt.title('XLOCAL')

                    plt.figure()
                    plt.imshow(s_out_rec_train[:, 0, selected_neuron_id, :], aspect='auto')
                    # plt.ylim([0, 128*2])
                    plt.title('INPUT NMF')
                    plt.xlabel('Time')
                    plt.ylabel('TrialxVariable')

                    # s_out_rec_train shape:  (trial x variable) x fanout x channels x time
                    print('s_out_train shape', s_out_rec_train.shape)
                    plt.figure()
                    plt.imshow(H[neuron_id].clone().detach(), aspect='auto',interpolation='nearest')
                    # plt.ylim([0, 128*2])
                    plt.title('H NMF')
                    plt.xlabel('Rank')
                    plt.ylabel('TrialxVariable')
                    plt.figure()
                    plt.imshow(coeff[:,None].clone().detach(), aspect='auto',interpolation='nearest')
                    # plt.ylim([0, 128*2])
                    plt.title('w classifier')
                    plt.xlabel('Rank')
                    plt.ylabel('TrialxVariable')
                    plt.figure()
                    plt.imshow(net.W.clone().detach(), aspect='auto')
                    # plt.ylim([0, 128*2])
                    plt.title('W NMF')
                    plt.xlabel('Time')
                    plt.ylabel('Rank')
                    # plt.figure()
                    # plt.imshow(net().clone().detach(), aspect='auto')
                    # # plt.ylim([0, 128*2])
                    # plt.title('OUT NMF')
                    # plt.xlabel('Time')
                    # plt.ylabel('TrialxVariable')
                    plt.figure()
                    plt.imshow(torch.concat(output_vs_label, dim=1), aspect='auto', cmap='seismic', interpolation='nearest')
                    plt.colorbar()
                    plt.title('OUTPUT vs LABEL')
                    # plt.xlabel('Output|Label')
                    plt.ylabel('TrialxVariable')
                    plt.xticks([0, 1], ['Output', 'Label'])
                    # plt.show()
                    print('eee macarena')

                loss = training['criterion'](outputs, label)
                # if torch.isnan(loss):
                print('loss',loss)
                # get gradients w.r.t to parameters
                loss.backward()
                list_loss.append(loss.clone().detach())
                # update parameters
                training['optimizer'].step()
                acc = ((outputs_norm - label_norm) ** 2).sum()/x_local.shape[0]
                print('acc', acc)

                # print('counter end train', counter)

            else:
                # print('testing xlocal mean',x_local.mean())
                # print('test total spikes for neuron', neuron_id, ':', V_matrix.sum())
                predicted = model(H[neuron_id])
                # plt.figure()
                # plt.imshow(H[neuron_id].clone().detach(),aspect = 'auto',interpolation='nearest')
                # plt.title('H[4]')
                # plt.figure()
                # plt.imshow(net.H.clone().detach(), aspect='auto',interpolation='nearest')
                # plt.title('net.H')
                # plt.figure()
                # plt.imshow(predicted.clone().detach(), aspect='auto',interpolation='nearest')
                # # print(model.parameters())
                # plt.show()
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
                    plt.imshow(net.H.clone().detach(), aspect='auto', interpolation='nearest')
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
                list_mi.append(mi)
    if len(training):
        return list_loss
    else:
        return list_mi


def MI_neuron_params(neuron_param_values, name_param_sweeped):

    # Set results folder:
    results_dir = set_results_folder(name_param_sweeped, exp_id)
    results_dir += '/'

    # Filename metadata:
    metadatafilename = results_dir + '/metadata.txt'

    # Create file with metadata
    addHeaderToMetadata(metadatafilename, 'Simulation')

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(0)

    cuda = torch.cuda.is_available()
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

    rank_NMF = int(params['rank_NMF'])
    range_weight_init = 10
    model = linearRegression(rank_NMF, 1)
    #model.linear.weight = torch.nn.Parameter(model.linear.weight*range_weight_init)

    coeff = [p for p in model.parameters()][0][0]
    coeff = coeff.clone().detach()

    #TODO: Check this
    if torch.cuda.is_available():
        pin_memory=True
    else:
        pin_memory=False

    # The log softmax function across output units
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory, generator=torch.Generator(device=device))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory, generator=torch.Generator(device=device))
    # pbar = trange(nb_epochs)
    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    optimizer = torch.optim.Adamax(model.parameters(), lr=learningRate)
    params['optimizer'] = optimizer.__class__.__name__
    print('Traning classifier with {} optimizer', params['optimizer'])

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
    for e in range(nb_epochs):
        local_loss = []
        H_train = []
        H_test = []
        accs = []  # accs: mean training accuracies for each batch
        print('Epoch', e)
        list_loss = sim_batch(dl_train, device, neuron, varying_element, rank_NMF, model,
                              {'optimizer':optimizer, 'criterion':criterion}, list_loss=list_loss, results_dir=results_dir)
        list_mi = sim_batch(dl_test, device, neuron, varying_element, rank_NMF, model, list_mi=list_mi, results_dir=results_dir)

    train_duration = time.time() - t_start
    addToNetMetadata(metadatafilename, 'sim duration (sec)', train_duration)

    list_mi = sim_batch(dl_test, device, neuron, varying_element, rank_NMF, model, list_mi=list_mi, final = True, results_dir=results_dir)
    fig = plt.figure()
    plt.plot(list_loss)
    plt.xlabel('Epochs x Trials')
    plt.ylabel('Loss')
    if save_out:
        fig.savefig(results_dir+'Loss.pdf', format='pdf')

    fig = plt.figure()
    plt.plot(list_mi)
    plt.xlabel('Epochs x Trials')
    plt.ylabel('MI')
    if save_out:
        fig.savefig(results_dir+'MI.pdf', format='pdf')

    # Store data:
    # Loss:
    with open(results_dir+'Loss.pickle', 'wb') as f:
        pickle.dump(list_loss, f)
    # Mi:
    with open(results_dir+'MI.pickle', 'wb') as f:
        pickle.dump(list_mi, f)

if __name__ == "__main__":

    for name_param in sweep_param_name:

        # Generate dictionary with parameter values:
        dict_keys = generate_dict(name_param, variable_range)

        # Run mutual information analysis
        MI_neuron_params(dict_keys, name_param)
