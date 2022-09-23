import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import trange

import numpy as np

from datasets import load_analog_data
from parameters.MN_params import MNparams_dict, INIT_MODE
from models import Encoder, LIF_neuron, MN_neuron
from auxiliary import compute_classification_accuracy, plot_spikes

def main():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###########################################
    ##              Parameters               ##
    ###########################################
    threshold = "enc"
    run = "_3"

    file_dir_params = 'parameters/'
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
    file_name = "data/data_braille_letters_digits.pkl"
    ds_train, ds_test, labels, nb_channels, data_steps = load_analog_data(file_name, upsample_fac)
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

    a = torch.empty((nb_inputs,))
    nn.init.normal_(
        a, mean=MNparams_dict[INIT_MODE][0], std=fwd_weight_scale / np.sqrt(nb_inputs))

    A1 = torch.empty((nb_inputs,))
    nn.init.normal_(
        A1, mean=MNparams_dict[INIT_MODE][1], std=fwd_weight_scale / np.sqrt(nb_inputs))

    A2 = torch.empty((nb_inputs,))
    nn.init.normal_(
        A2, mean=MNparams_dict[INIT_MODE][2], std=fwd_weight_scale / np.sqrt(nb_inputs))

    network = nn.Sequential(Encoder(nb_inputs, encoder_weight_scale, nb_input_copies),
                            MN_neuron(nb_inputs, a, A1, A2, train=True),
                            LIF_neuron(nb_inputs, nb_hidden, alpha, beta, is_recurrent=True,
                                    fwd_weight_scale=fwd_weight_scale, rec_weight_scale=rec_weight_scale),
                            LIF_neuron(nb_hidden, nb_outputs, alpha, beta, is_recurrent=False, fwd_weight_scale=fwd_weight_scale, rec_weight_scale=rec_weight_scale)).to(device)
    print(network)

    ###########################################
    ##               Training                ##
    ###########################################
    batch_size = 128

    optimizer = torch.optim.Adamax(
        network.parameters(), lr=0.005, betas=(0.9, 0.995))

    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

    ttc_hist = []
    loss_hist = []
    accs_hist = [[], []]

    writer = SummaryWriter()  # For logging purpose

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    pbar = trange(nb_epochs)
    for e in pbar:
        local_loss = []
        accs = []  # accs: mean training accuracies for each batch
        for x_local, y_local in dl_train:
            x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)

            # Reset all the layers in the network
            for layer in network:
                if hasattr(layer.__class__, 'reset'):
                    layer.reset()

            # Simulate the network
            # we are going to record the hidden layer
            # spikes for regularization purposes
            loss_local = 0
            spk_rec = []
            out_rec = []
            s_out_rec = []
            for t in range(nb_steps):
                out = network(x_local[:, t])

                # Get the spikes of the hidden layer
                spk_rec.append(network[-2].state.S)
                # Get the voltage of the last layer
                out_rec.append(network[-1].state.mem)
                s_out_rec.append(out)
            spk_rec = torch.stack(spk_rec, dim=1)
            out_rec = torch.stack(out_rec, dim=1)
            s_out_rec = torch.stack(s_out_rec, dim=1)

            m = torch.sum(s_out_rec, 1)  # sum over time
            log_p_y = log_softmax_fn(m)

            # Here we can set up our regularizer loss
            reg_loss = params['reg_spikes'] * torch.mean(
                torch.sum(spk_rec, 1))  # e.g., L1 loss on total number of spikes (original: 1e-3)
            reg_loss += params['reg_neurons'] * torch.mean(
                torch.sum(torch.sum(spk_rec, dim=0), dim=0) ** 2)  # L2 loss on spikes per neuron (original: 2e-6)

            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local) + reg_loss

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())

            # compare to labels
            _, am = torch.max(m, 1)  # argmax over output units
            tmp = np.mean((y_local == am).detach().cpu().numpy())
            accs.append(tmp)

        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)

        # mean_accs: mean training accuracy of current epoch (average over all batches)
        mean_accs = np.mean(accs)
        accs_hist[0].append(mean_accs)

        # Calculate test accuracy in each epoch on the testing dataset
        test_acc, test_ttc, spk_hidden, spk_output = compute_classification_accuracy(
            params, dl_test, network, True, device)
        accs_hist[1].append(test_acc)  # only safe best test
        ttc_hist.append(test_ttc)

        ###########################################
        ##               Plotting                ##
        ###########################################

        fig1 = plot_spikes(spk_hidden)
        fig2 = plot_spikes(spk_output)

        ###########################################
        ##                Logging                ##
        ###########################################

        writer.add_scalar('Accuracy/test', test_acc, global_step=e)
        writer.add_scalar('Accuracy/train', mean_accs, global_step=e)
        writer.add_scalar('Loss', mean_loss, global_step=e)
        writer.add_figure('Hidden spikes', fig1, global_step=e)
        writer.add_figure('Output spikes', fig2, global_step=e)

        pbar.set_postfix_str("Train accuracy: " + str(np.round(accs_hist[0][-1] * 100, 2)) + '%. Test accuracy: ' + str(
            np.round(accs_hist[1][-1] * 100, 2)) + '%, Loss: ' + str(np.round(mean_loss, 2)))

if __name__ == "__main__":
    main()