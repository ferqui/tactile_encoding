import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchnmf.nmf import NMF
from torchnmf.metrics import kl_div
import matplotlib.pyplot as plt

from tqdm import trange

import numpy as np

import models
from datasets import load_analog_data
from parameters.MN_params import MNparams_dict, INIT_MODE
from models import Encoder, LIF_neuron, MN_neuron_IT
from auxiliary import compute_classification_accuracy, plot_spikes

from torch.autograd import Variable
torch.manual_seed(0)



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

def sim_batch(dataset, device, neuron, varying_element, rank_NMF, model, training ={}, list_loss=[]):
    #training has optimizer,criterion
    counter = 0

    for x_local, y_local in dataset:
        # print('aaaa')
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
        #     print(i*step_t)
        # ## s_out_rec_train shape: trial x time x fanout x variable x channels

        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))

        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        label = torch.ones([s_out_rec_train.shape[1], varying_element.shape[0]]) * varying_element
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        label = label.T.flatten()[:, None]
        label_min = label.min()
        label_diff = label.max()-label.min()
        label_norm = (label - label_min)/(label_diff)
        print('label_norm min max',label_norm.min(),label_norm.max())

        # plt.show()

        H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        selected_neuron_id = 4
        for neuron_id in [selected_neuron_id]:
            V_matrix = s_out_rec_train[:, 0, neuron_id, :]

            net = NMF(V_matrix.shape, rank=rank_NMF)
            net.fit(V_matrix)

            H[neuron_id] = net.H


            import scipy.spatial as sp
            cdist = 1 - sp.distance.cdist(net().clone().detach(), V_matrix.clone().detach(), 'cosine')
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
                if counter == 100000:

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
                    plt.imshow(net.H.clone().detach(), aspect='auto')
                    # plt.ylim([0, 128*2])
                    plt.title('H NMF')
                    plt.xlabel('Time')
                    plt.ylabel('TrialxVariable')
                    plt.figure()
                    plt.imshow(net().clone().detach(), aspect='auto')
                    # plt.ylim([0, 128*2])
                    plt.title('OUT NMF')
                    plt.xlabel('Time')
                    plt.ylabel('TrialxVariable')
                    plt.figure()
                    plt.imshow(torch.concat(output_vs_label, dim=1), aspect='auto', cmap='seismic', interpolation='nearest')
                    plt.colorbar()
                    plt.title('OUTPUT vs LABEL')
                    # plt.xlabel('Output|Label')
                    plt.ylabel('TrialxVariable')
                    plt.xticks([0, 1], ['Output', 'Label'])
                    plt.show()
                    print('eee macarena')
                loss = training['criterion'](outputs_norm, label_norm)
                # if torch.isnan(loss):
                print('loss',loss)
                # get gradients w.r.t to parameters
                loss.backward()
                list_loss.append(loss.clone().detach())
                # update parameters
                training['optimizer'].step()
                acc = ((outputs_norm - label_norm) ** 2).sum()/x_local.shape[0]
                print('acc', acc)

            else:
                # print('testing xlocal mean',x_local.mean())
                # print('test total spikes for neuron', neuron_id, ':', V_matrix.sum())
                predicted = model(H[neuron_id])
                outputs_norm = (predicted - label_min) / (label_diff)

                acc = ((outputs_norm - label_norm) ** 2).sum()/x_local.shape[0]
                print('acc', acc)

    return list_loss

def main():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(0)

    ###########################################
    ##              Parameters               ##
    ###########################################
    threshold = "enc"
    run = "_3"

    file_dir_params = '../parameters/'
    param_filename = 'parameters_th' + str(threshold)
    file_name_parameters = file_dir_params + param_filename + '.txt'
    params = {}
    with open(file_name_parameters) as file:
        for line in file:
            (key, value) = line.split()
            if key == 'time_bin_size' or key == 'nb_input_copies':
                params[key] = int(value)
            else:
                params[key] = np.double(value)

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
    nb_outputs = len(np.unique(params['labels']))

    # Learning parameters
    nb_steps = params['data_steps']
    nb_epochs = 300

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
    n_param_values = 10
    a = torch.Tensor(np.linspace(-10, 10, n_param_values))
    # nn.init.normal_(
    #     a, mean=MNparams_dict['A2B'][0], std=fwd_weight_scale / np.sqrt(nb_inputs))

    A1 = torch.Tensor(np.linspace(0, 0, n_param_values))

    A2 = torch.Tensor(np.linspace(0, 0, n_param_values))


    fanout = 1  # number of output neurons from the linear expansion
    neuron = models.MN_neuron_IT(params['nb_channels'], fanout, n_param_values, a, A1, A2, train=False)

    batch_size = 50

    inputDim = 1  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    learningRate = 0.1#10#0.01
    epochs = 100
    rank_NMF = 10
    #model = linearRegression(rank_NMF, 1)
    model = linearRegression(rank_NMF,1)

    coeff = [p for p in model.parameters()][0][0]
    coeff = coeff.clone().detach()
    # plt.figure()
    # plt.hist(coeff)
    # plt.show()

    # The log softmax function across output units
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # pbar = trange(nb_epochs)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    list_loss = []
    for e in range(100):
        local_loss = []
        H_train = []
        H_test = []
        accs = []  # accs: mean training accuracies for each batch
        counter = 0
        list_loss = sim_batch(dl_train, device, neuron, varying_element, rank_NMF, model, {'optimizer':optimizer, 'criterion':criterion}, list_loss)
        _ = sim_batch(dl_test, device, neuron, varying_element, rank_NMF, model)

        print('Hiiiiiiiiiiii')
    plt.plot(list_loss)
    plt.show()

    ## last test
    try:
        for x_local, y_local in dl_test:
            x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)

            # Reset all the layers in the network
            neuron.reset()
            s_out_rec = []
            for t in range(nb_steps):
                out = neuron(x_local[:, t, None, None])
                s_out_rec.append(out)
            s_out_rec_test = torch.stack(s_out_rec, dim=1)
            # print('s_out_rec_test shape',s_out_rec_test.shape)
            ### shape: trial x time x fanout x variable x channels
            s_out_rec_test = torch.permute(s_out_rec_test, (0, 3, 2, 4, 1))
            ### shape:  trial x fanout x channels x time x variable
            label = label.T.flatten()[:, None]
            # print('label shape',label.shape)

            s_out_rec_test = torch.flatten(s_out_rec_test, start_dim=0, end_dim=1)
            ### shape:  trial x fanout x channels x (time x variable)

            label = torch.ones([s_out_rec_test.shape[0], varying_element.shape[0]]) * varying_element

            H_test = torch.zeros([s_out_rec_test.shape[2], s_out_rec_test.shape[0], rank_NMF])
            # print('H_test shape',H_test.shape)
            label_unique = torch.unique(label)
            pdf_x1x2 = torch.zeros([len(label_unique), len(label_unique)])
            for neuron_id in range(1):
                V_matrix = s_out_rec_test[:, 0, neuron_id, :]
                net = NMF(V_matrix.shape, rank=rank_NMF)
                net.fit(V_matrix)
                H_test[neuron_id] = net.H
                predicted = model(H_test[neuron_id])
                for pred in predicted:
                    for lab in labels:
                        lab_pos = torch.where(label_unique == lab)
                        pred_pos = torch.where(label_unique == pred)
                        pdf_x1x2[lab_pos, pred_pos] += 1
                pdf_x1x2 = pdf_x1x2 / torch.sum(pdf_x1x2)
                pdf_x1 = torch.sum(pdf_x1x2, dim=0)
                pdf_x2 = torch.sum(pdf_x1x2, dim=1)
                mi = torch.zeros(1)
                for el1_idx, pdf_x1_el in enumerate(pdf_x1):
                    for el2_idx, pdf_x2_el in enumerate(pdf_x2):
                        mi += pdf_x1x2[el1_idx, el2_idx] * torch.log(pdf_x1x2[el1_idx, el2_idx] / (pdf_x1_el * pdf_x2_el))
                print('mutual information', mi)
    except:
        print('Hello')

if __name__ == "__main__":
    main()
