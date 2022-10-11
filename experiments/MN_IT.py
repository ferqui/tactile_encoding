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

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

def main():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    ds_train, ds_test, labels, nb_channels, data_steps = load_analog_data(file_name, upsample_fac, specify_letters = ['A'])
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
    alpha = float(np.exp(-params['data_steps']*0.001 / tau_syn))
    beta = float(np.exp(-params['data_steps']*0.001 / tau_mem))

    encoder_weight_scale = 1.0
    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = params['weight_scale_factor'] * fwd_weight_scale

    ###########################################
    ##                Network                ##
    ###########################################

    # a = torch.empty((nb_inputs,))
    a = torch.Tensor(np.linspace(-100,100,10))
    # nn.init.normal_(
    #     a, mean=MNparams_dict['A2B'][0], std=fwd_weight_scale / np.sqrt(nb_inputs))

    A1 = torch.Tensor(np.linspace(0,0,10))

    A2 = torch.Tensor(np.linspace(0,0,10))

    varying_element = a

    neuron = models.MN_neuron_IT(params['nb_channels'],1,10,a,A1,A2,train=False)

    batch_size = 128

    inputDim = 1  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    learningRate = 0.01
    epochs = 100
    rank_NMF = 10
    model = linearRegression(10,1)

    # The log softmax function across output units
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    pbar = trange(nb_epochs)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    for e in range(300):
        local_loss = []
        H_train = []
        H_test = []
        accs = []  # accs: mean training accuracies for each batch
        for x_local, y_local in dl_train:
            # print('aaaa')
            x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)

            # Reset all the layers in the network
            neuron.reset()

            s_out_rec = []
            for t in range(nb_steps):
                out = neuron(x_local[:, t,None,None])
                s_out_rec.append(out)


            s_out_rec_train = torch.stack(s_out_rec, dim=1)
            ### s_out_rec_train shape: trial x time x fanout x variable x channels
            s_out_rec_train = torch.permute(s_out_rec_train,(0,3,2,4,1))
            ### s_out_rec_train shape:  trial x variable x fanout x channels x time
            label = torch.ones([s_out_rec_train.shape[0], varying_element.shape[0]])*varying_element
            s_out_rec_train = torch.flatten(s_out_rec_train, start_dim = 0,end_dim = 1)
            label = label.T.flatten()[:,None]

            ### s_out_rec_train shape:  (trial x variable) x fanout x channels x time
            # print('s_out_train shape',s_out_rec_train.shape)


            H_train = torch.zeros([s_out_rec_train.shape[2],s_out_rec_train.shape[0],rank_NMF])
            for neuron_id in range(1):
                V_matrix = s_out_rec_train[:,0,neuron_id,:]

                net = NMF(V_matrix.shape, rank=rank_NMF)
                net.fit(V_matrix)
                H_train[neuron_id] = net.H

                optimizer.zero_grad()
                outputs = model(H_train[neuron_id])
                loss = criterion(torch.round(outputs.type(torch.float)), torch.round(label.type(torch.float)))
                # print(torch.round(outputs.type(torch.float)))
                # print(torch.round(label.type(torch.float)))
                print(loss)
                # get gradients w.r.t to parameters
                loss.backward()

                # update parameters
                optimizer.step()

        for x_local, y_local in dl_test:
            x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)

            # Reset all the layers in the network
            neuron.reset()
            s_out_rec = []
            for t in range(nb_steps):
                out = neuron(x_local[:, t,None,None])
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

            for neuron_id in range(1):
                V_matrix = s_out_rec_test[:, 0, neuron_id, :]
                net = NMF(V_matrix.shape, rank=rank_NMF)
                net.fit(V_matrix)
                H_test[neuron_id] = net.H
                predicted = model(H_test[neuron_id])
                # print('predicted shape', predicted.shape)
                # print(predicted.type(torch.int32))
                # print(label.type(torch.int32))
                tmp = np.mean((predicted.type(torch.int32) == label.type(torch.int32)).detach().cpu().numpy())
                print('acc',tmp)

            print('loss {}, ac'.format(loss.item(),tmp.item()))
    ## last test
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
        pdf_x1x2 = torch.zeros([len(label_unique),len(label_unique)])
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
                    pdf_x1x2[lab_pos,pred_pos] += 1
            pdf_x1x2 = pdf_x1x2/torch.sum(pdf_x1x2)

if __name__ == "__main__":
    main()