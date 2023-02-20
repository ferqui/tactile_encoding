"""
*** add description ***

Fra, Vittorio,
Politecnico di Torino,
EDA Group,
Torino, Italy.

Muller-Cleve, Simon F.,
Istituto Italiano di Tecnologia - IIT,
Event-driven perception in robotics - EDPR,
Genova, Italy.
"""

import logging
import argparse
import numpy as np
import pandas as pd

import nni
from nni.tools.nnictl import updater

import os
import datetime
from subprocess import check_output
import random

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#from tactile_encoding.utils.utils import check_cuda, value2index, create_directory, load_layers
from utils.utils import check_cuda, create_directory


exp_name = "train_spike_classifier" # name of the experiment as in the "main" script for NNI configuration
LOG = logging.getLogger(exp_name)

searchspace_filename = "{}_searchspace".format(exp_name)
searchspace_path = "./searchspaces/{}.json".format(searchspace_filename)
#with open(searchspace_path, "r") as read_searchspace:
#    search_space = json.load(read_searchspace)

# set up CUDA device
manual_selection = True
if not manual_selection:
    if torch.cuda.is_available():
        gpu_query = str(check_output(["nvidia-smi", "--format=csv", "--query-gpu=index"]), 'utf-8').splitlines()
        gpu_devices = [int(ii) for ii in gpu_query if ii != 'index']
        gpu_idx = random.choice(gpu_devices)
else:
    gpu_idx = 0
global device
device = check_cuda(gpu_sel=gpu_idx, gpu_mem_frac=0.3)

global use_seed
use_seed = False

global seed
if use_seed:
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    LOG.debug(print("Seed set to {}".format(seed)))
else:
    seed = None


def run_snn(inputs, layers):

    if use_trainable_out and use_trainable_tc:
        w1, w2, v1, alpha1, beta1, alpha2, beta2, out_scale, out_offset = layers
    elif use_trainable_tc:
        w1, w2, v1, alpha1, beta1, alpha2, beta2 = layers
    elif use_trainable_out:
        w1, w2, v1, out_scale, out_offset = layers
    else:
        w1, w2, v1 = layers
    if use_dropout:
        # using dropout on (n in %)/100 of spikes
        dropout = nn.Dropout(p=0.25)
    if use_trainable_tc:
        alpha1, beta1 = torch.abs(alpha1), torch.abs(beta1)
        alpha2, beta2 = torch.abs(alpha2), torch.abs(beta2)

    bs = inputs.shape[0]

    h1 = torch.einsum(
        "abc,cd->abd", (inputs.tile((nb_input_copies,)), w1))
    # h1 = torch.einsum(
    #     "abc,cd->abd", (inputs, w1))
    if use_dropout:
        h1 = dropout(h1)
    if use_trainable_tc:
        spk_rec, mem_rec = recurrent_layer.compute_activity_tc(
            bs, nb_hidden, h1, v1, alpha1, beta1, nb_steps)
    else:
        spk_rec, mem_rec = recurrent_layer.compute_activity(
            bs, nb_hidden, h1, v1, nb_steps)

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))
    if use_dropout:
        h2 = dropout(h2)
    if use_trainable_tc:
        s_out_rec, out_rec = feedforward_layer.compute_activity_tc(
            bs, nb_outputs, h2, alpha2, beta2, nb_steps)
    else:
        s_out_rec, out_rec = feedforward_layer.compute_activity(
            bs, nb_outputs, h2, nb_steps)

    if use_trainable_out:
        # trainable output spike scaling
        # mean_firing_rate = torch.div(torch.sum(s_out_rec,1), s_out_rec.shape[1]) # mean firing rate
        # s_out_rec = mean_firing_rate*layers[5] + layers[6]
        s_out_rec = torch.sum(s_out_rec, 1)*out_scale + \
            out_offset  # sum spikes

    other_recs = [mem_rec, spk_rec, out_rec]
    layers_update = layers

    return s_out_rec, other_recs, layers_update


def nni_train(dataset, lr=0.0015, nb_epochs=300, opt_parameters=None, layers=None, dataset_val=None):

    if (opt_parameters != None) & (layers != None):
        parameters = opt_parameters  # The paramters we want to optimize
        layers = layers
    elif (opt_parameters != None) & (layers == None):
        parameters = opt_parameters
        if use_trainable_out and use_trainable_tc:
            layers = [w1, w2, v1, alpha1, beta1,
                      alpha2, out_scale, out_offset]
        elif use_trainable_out:
            layers = [w1, w2, v1, out_scale, out_offset]
        elif use_trainable_tc:
            layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
        else:
            layers = [w1, w2, v1]
    elif (opt_parameters == None) & (layers != None):
        if use_trainable_out and use_trainable_tc:
            layers = [w1, w2, v1, alpha1, beta1, alpha2,
                      beta2, out_scale, out_offset]
        elif use_trainable_out:
            layers = [w1, w2, v1, out_scale, out_offset]
        elif use_trainable_tc:
            layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
        else:
            layers = [w1, w2, v1]
        layers = layers
    elif (opt_parameters == None) & (layers == None):
        if use_trainable_out and use_trainable_tc:
            parameters = [w1, w2, v1, alpha1, beta1, alpha2,
                          beta2, out_scale, out_offset]
            layers = [w1, w2, v1, alpha1, beta1, alpha2,
                      beta2, out_scale, out_offset]
        elif use_trainable_out:
            parameters = [w1, w2, v1, out_scale, out_offset]
            layers = [w1, w2, v1, out_scale, out_offset]
        elif use_trainable_tc:
            parameters = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
            layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
        else:
            parameters = [w1, w2, v1]
            layers = [w1, w2, v1]

    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

    g = torch.Generator()
    #g.manual_seed(seed)
    #generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
    #                       num_workers=0, pin_memory=True, generator=g)
    #generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
    #                       num_workers=4, pin_memory=True, generator=g)
    generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) # pin_memroy with CPU tensors only

    # The optimization loop
    loss_hist = [[], []]
    accs_hist = [[], []]
    for e in range(nb_epochs):
        ### "debug" print:
        #logging.info(f"Starting epoch {e+1} of {nb_epochs} ")
        ###
        # learning rate decreases over epochs
        optimizer = torch.optim.Adamax(
            parameters, lr=lr, betas=(0.9, 0.995))
        # if e > nb_epochs*25:
        #     lr = lr * 0.98
        local_loss = []
        # accs: mean training accuracies for each batch
        accs = []
        for x_local, y_local in generator:
            x_local, y_local = x_local.to(
                device, non_blocking=True), y_local.to(device, non_blocking=True)
            spks_out, recs, layers_update = run_snn(x_local, layers)
            # [mem_rec, spk_rec, out_rec]
            _, spk_rec, _ = recs

            # with output spikes
            if use_trainable_out:
                m = spks_out
            else:
                m = torch.sum(spks_out, 1)  # sum over time

            # cross entropy loss on the active read-out layer
            log_p_y = log_softmax_fn(m)

            # # L1 loss on total number of spikes (hidden layer)
            # reg_loss = 1e-3**torch.mean(torch.sum(spk_rec, 1))
            # # L2 loss on spikes per neuron (hidden layer)
            # reg_loss += 1e-6* \
            #     torch.mean(torch.sum(torch.sum(spk_rec, dim=0), dim=0)**2)
            # L1 loss on total number of spikes (hidden layer)
            reg_loss = 1e-4*torch.mean(torch.sum(spk_rec, 1))
            # L2 loss on spikes per neuron (hidden layer)
            reg_loss = reg_loss + 1e-8 * \
                torch.mean(torch.sum(torch.sum(spk_rec, dim=0), dim=0)**2)

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
        loss_hist[0].append(mean_loss)

        # mean_accs: mean training accuracy of current epoch (average over all batches)
        mean_accs = np.mean(accs)
        accs_hist[0].append(mean_accs)

        # Calculate validation (test) accuracy in each epoch
        if dataset_val is not None:
            val_acc, val_loss = compute_classification_accuracy(
                dataset_val,
                layers=layers_update
            )
            # only safe best validation (test)
            accs_hist[1].append(val_acc)
            # only safe loss of best validation (test)
            loss_hist[1].append(val_loss)

        if dataset_val is None:
            # save best training
            if mean_accs >= np.max(accs_hist[0]):
                best_acc_layers = []
                for ii in layers_update:
                    best_acc_layers.append(ii.detach().clone())
        else:
            # save best validation (test)
            if val_acc >= np.max(accs_hist[1]):
                best_acc_layers = []
                for ii in layers_update:
                    best_acc_layers.append(ii.detach().clone())

        LOG.debug(print("Epoch {}/{} done. Training accuracy (loss): {:.2f}% ({:.5f}), Validation accuracy (loss): {:.2f}% ({:.5f}).".format(
            e + 1, nb_epochs, accs_hist[0][-1]*100, loss_hist[0][-1], accs_hist[1][-1]*100, loss_hist[1][-1])))
        
        nni.report_intermediate_result({"default": np.round(accs_hist[1][-1]*100,4),
                                        "training": np.round(accs_hist[0][-1]*100,4)})

    return loss_hist, accs_hist, best_acc_layers


def nni_build_and_train(params, ds_train, ds_val, epochs=300):

    data_steps = len(next(iter(ds_train))[0])

    global nb_input_copies
    # Num of spiking neurons used to encode each channel
    nb_input_copies = 1  # params['nb_input_copies']

    # Network parameters
    global nb_inputs
    nb_inputs = 1  # 24*nb_input_copies
    global nb_outputs
    nb_outputs = 20  # number of spiking behaviours from MN paper
    global nb_hidden
    nb_hidden = int(params["nb_hidden"])
    global nb_steps
    nb_steps = data_steps

    tau_mem = params["tau_mem"]
    tau_syn = params["tau_syn"]

    if not use_trainable_tc:
        global alpha
        global beta
    dt = 1e-3  # ms
    alpha = torch.as_tensor(float(np.exp(-dt/tau_syn)))
    beta = torch.as_tensor(float(np.exp(-dt/tau_mem)))

    fwd_weight_scale = params["fwd_weights_std"]
    rec_weight_scale = params["rec_weights_std"]

    # Spiking network
    layers = []

    # recurrent layer
    w1, v1 = recurrent_layer.create_layer(
        nb_inputs, nb_hidden, fwd_weight_scale, rec_weight_scale)

    # readout layer
    w2 = feedforward_layer.create_layer(
        nb_hidden, nb_outputs, fwd_weight_scale)

    if use_trainable_tc:
        # time constants
        alpha1, beta1 = trainable_time_constants.create_time_constants(
            nb_hidden, alpha, beta, use_trainable_tc)

        alpha2, beta2 = trainable_time_constants.create_time_constants(
            nb_outputs, alpha, beta, use_trainable_tc)

    layers.append(w1), layers.append(w2), layers.append(v1)
    if use_trainable_tc:
        layers.append(alpha1), layers.append(
            beta1), layers.append(alpha2), layers.append(beta2)

    if use_trainable_out:
        # include trainable output for readout layer (linear: y = out_scale * x + out_offset)
        out_scale = torch.empty(
            (nb_outputs),  device=device, dtype=torch.float, requires_grad=True)
        torch.nn.init.ones_(out_scale)
        layers.append(out_scale)
        out_offset = torch.empty(
            (nb_outputs),  device=device, dtype=torch.float, requires_grad=True)
        torch.nn.init.zeros_(out_offset)
        layers.append(out_offset)

    layers_init = []
    for ii in layers:
        layers_init.append(ii.detach().clone())

    if use_trainable_out and use_trainable_tc:
        opt_parameters = [w1, w2, v1, alpha1, beta1,
                          alpha2, beta2, out_scale, out_offset]
    elif use_trainable_tc:
        opt_parameters = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
    elif use_trainable_out:
        opt_parameters = [w1, w2, v1, out_scale, out_offset]
    else:
        opt_parameters = [w1, w2, v1]

    # a fixed learning rate is already defined within the train function, that's why here it is omitted
    loss_hist, accs_hist, best_layers = nni_train(
        ds_train, lr=lr, nb_epochs=epochs, opt_parameters=opt_parameters, layers=layers, dataset_val=ds_val)

    # best training and validation (test) at best training
    acc_best_train = np.max(accs_hist[0])  # returns max value
    acc_best_train = acc_best_train*100
    idx_best_train = np.argmax(accs_hist[0])  # returns index of max value
    acc_val_at_best_train = accs_hist[1][idx_best_train]*100

    # best validation (test) and training at best validation (test)
    acc_best_val = np.max(accs_hist[1])
    acc_best_val = acc_best_val*100
    idx_best_val = np.argmax(accs_hist[1])
    acc_train_at_best_val = accs_hist[0][idx_best_val]*100

    LOG.debug(print(
        "\n------------------------------------------------------------------------------------"))
    logging.info("Final results: ")
    LOG.debug(print("Best training accuracy: {:.2f}% and according validation accuracy: {:.2f}% at epoch: {}".format(
        acc_best_train, acc_val_at_best_train, idx_best_train+1)))
    LOG.debug(print("Best validation accuracy: {:.2f}% and according train accuracy: {:.2f}% at epoch: {}".format(
        acc_best_val, acc_train_at_best_val, idx_best_val+1)))
    LOG.debug(print(
        "------------------------------------------------------------------------------------"))
    LOG.debug(print(
            "------------------------------------------------------------------------------------\n"))
    return loss_hist, accs_hist, best_layers


def compute_classification_accuracy(dataset, layers=None, label_probabilities=False, shuffle=False):
    """ Computes classification accuracy on supplied data in batches. """

    # generator = DataLoader(dataset, batch_size=batch_size,
    #                     shuffle=False, num_workers=4, pin_memory=True)
    generator = DataLoader(dataset, batch_size=batch_size,
                           shuffle=shuffle, num_workers=0) # pin_memory with CPU tensors only
    accs = []
    losss = []
    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

    for x_local, y_local in generator:
        x_local, y_local = x_local.to(
            device, non_blocking=True), y_local.to(device, non_blocking=True)
        if layers == None:
            if use_trainable_out and use_trainable_tc:
                layers = [w1, w2, v1, alpha1, beta1, alpha2,
                          beta2, out_scale, out_offset]
            elif use_trainable_out:
                layers = [w1, w2, v1, out_scale, out_offset]
            elif use_trainable_tc:
                layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
            else:
                layers = [w1, w2, v1]
            spks_out, _, _ = run_snn(x_local, layers)
        else:
            spks_out, _, _ = run_snn(x_local, layers)
        # with output spikes
        if use_trainable_out:
            m = spks_out
        else:
            m = torch.sum(spks_out, 1)  # sum over time
        _, am = torch.max(m, 1)     # argmax over output units
        # compute validation (test) loss
        log_p_y = log_softmax_fn(m)
        loss_val = loss_fn(log_p_y, y_local).detach().cpu().numpy()
        losss.append(loss_val)
        # compute acc
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)

        if label_probabilities:
            return np.mean(accs), np.mean(losss), torch.exp(log_p_y)
        else:
            return np.mean(accs), np.mean(losss)


def ConfusionMatrix(dataset, save, layers=None, labels=None, use_seed=use_seed):
        
    if use_seed:
        g = torch.Generator()
        g.manual_seed(seed)
        # generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
        #                     num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        #generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
        #                       num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, generator=g) # pin_memory with CPU tensors only
    else:
        generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) # pin_memory with CPU tensors only
        
    accs = []
    trues = []
    preds = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(
            device, non_blocking=True), y_local.to(device, non_blocking=True)
        if layers == None:
            if use_trainable_out and use_trainable_tc:
                layers = [w1, w2, v1, alpha1, beta1, alpha2,
                          beta2, out_scale, out_offset]
            elif use_trainable_out:
                layers = [w1, w2, v1, out_scale, out_offset]
            elif use_trainable_tc:
                layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
            else:
                layers = [w1, w2, v1]
            spks_out, _, _ = run_snn(x_local, layers)
        else:
            spks_out, _, _ = run_snn(x_local, layers)
        # with output spikes
        if use_trainable_out:
            m = spks_out
        else:
            m = torch.sum(spks_out, 1)  # sum over time
        _, am = torch.max(m, 1)     # argmax over output units
        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
        trues.extend(y_local.detach().cpu().numpy())
        preds.extend(am.detach().cpu().numpy())

    logging.info("Accuracy from confusion matrix: {:.2f}% +- {:.2f}%".format(np.mean(accs)
                                                                      * 100, np.std(accs)*100))

    cm = confusion_matrix(trues, preds, normalize='true')
    cm_df = pd.DataFrame(cm, index=[ii for ii in labels], columns=[
        jj for jj in labels])
    plt.figure("cm", figsize=(12, 9))
    sn.heatmap(cm_df,
               annot=True,
               fmt='.1g',
               cbar=False,
               square=False,
               cmap="YlGnBu")
    plt.xlabel('\nPredicted')
    plt.ylabel('True\n')
    plt.xticks(rotation=0)
    if save:
        #path_to_save_fig = f'{path_for_plots}/generation_{generation+1}_individual_{best_individual+1}'
        path_to_save_fig = f'{path_for_plots}/cm'
        if use_trainable_tc:
            path_to_save_fig = f'{path_to_save_fig}_train_tc'
        if use_trainable_out:
            path_to_save_fig = f'{path_to_save_fig}_train_out'
        if use_dropout:
            path_to_save_fig = f'{path_to_save_fig}_dropout'
        #path_to_save_fig = f'{path_to_save_fig}_cm.png'
        path_to_save_fig = f'{path_to_save_fig}.png'
        plt.savefig(path_to_save_fig, dpi=300)
        plt.close()
    else:
        plt.show()


def NetworkActivity(dataset, save, layers=None, labels=None):

    g = torch.Generator()
    g.manual_seed(seed)
    # generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
    #                     num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, worker_init_fn=seed_worker, generator=g) # pin_memory with CPU tensors only

    accs = []
    trues = []
    preds = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(
            device, non_blocking=True), y_local.to(device, non_blocking=True)
        if layers == None:
            if use_trainable_out and use_trainable_tc:
                layers = [w1, w2, v1, alpha1, beta1, alpha2,
                          beta2, out_scale, out_offset]
            elif use_trainable_out:
                layers = [w1, w2, v1, out_scale, out_offset]
            elif use_trainable_tc:
                layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
            else:
                layers = [w1, w2, v1]
            spks_out, recs, _ = run_snn(x_local, layers)
        else:
            spks_out, recs, _ = run_snn(x_local, layers)

        # [mem_rec, spk_rec, out_rec]
        _, spk_rec, _ = recs

    nb_plt = 4
    gs = GridSpec(1, nb_plt)

    # hidden layer
    fig = plt.figure("hidden layer", figsize=(8, 6), dpi=300)
    plt.title("Hidden layer 1")
    for i in range(nb_plt):
        plt.subplot(gs[i])
        plt.imshow(spk_rec[i].detach().cpu().numpy().T,
                   cmap=plt.cm.gray_r, origin="lower")
        if i == 0:
            plt.xlabel("Time")
            plt.ylabel("Units")
        sn.despine()
    if save:
        path_to_save_fig = f'{path_for_plots}/generation_{generation+1}_individual_{best_individual+1}'
        if use_trainable_tc:
            path_to_save_fig = f'{path_to_save_fig}_train_tc'
        if use_trainable_out:
            path_to_save_fig = f'{path_to_save_fig}_train_out'
        if use_dropout:
            path_to_save_fig = f'{path_to_save_fig}_dropout'
        path_to_save_fig = f'{path_to_save_fig}_hidden_layer.png'
        plt.savefig(path_to_save_fig, dpi=300)
        plt.close()

    # output layer
    fig = plt.figure("output layer", figsize=(8, 6), dpi=300)
    plt.title("Output layer")
    for i in range(nb_plt):
        plt.subplot(gs[i])
        plt.imshow(spks_out[i].detach().cpu().numpy().T,
                   cmap=plt.cm.gray_r, origin="lower")
        if i == 0:
            plt.xlabel("Time")
            plt.ylabel("Units")
        sn.despine()
    if save:
        path_to_save_fig = f'{path_for_plots}/generation_{generation+1}_individual_{best_individual+1}'
        if use_trainable_tc:
            path_to_save_fig = f'{path_to_save_fig}_train_tc'
        if use_trainable_out:
            path_to_save_fig = f'{path_to_save_fig}_train_out'
        if use_dropout:
            path_to_save_fig = f'{path_to_save_fig}_dropout'
        path_to_save_fig = f'{path_to_save_fig}_output_layer.png'
        plt.savefig(path_to_save_fig, dpi=300)
        plt.close()
    else:
        plt.show()


class feedforward_layer:
    '''
    class to initialize and compute spiking feedforward layer
    '''
    def create_layer(nb_inputs, nb_outputs, scale):
        ff_layer = torch.empty(
            (nb_inputs, nb_outputs),  device=device, dtype=torch.float, requires_grad=True)
        torch.nn.init.normal_(ff_layer, mean=0.0,
                              std=scale/np.sqrt(nb_inputs))
        return ff_layer

    def compute_activity(nb_input, nb_neurons, input_activity, nb_steps):
        syn = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=torch.float)
        mem = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=torch.float)
        out = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=torch.float)
        mem_rec = []
        spk_rec = []

        # Compute feedforward layer activity
        for t in range(nb_steps):
            mthr = mem-1.0
            out = spike_fn(mthr)
            rst_out = out.detach()

            new_syn = alpha*syn + input_activity[:, t]
            new_mem = (beta*mem + syn)*(1.0-rst_out)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        # Now we merge the recorded membrane potentials into a single as_tensor
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        return spk_rec, mem_rec

    def compute_activity_tc(nb_input, nb_neurons, input_activity, alpha, beta, nb_steps):
        syn = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=torch.float)
        mem = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=torch.float)
        out = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=torch.float)
        mem_rec = []
        spk_rec = []

        # Compute feedforward layer activity
        for t in range(nb_steps):
            mthr = mem-1.0
            out = spike_fn(mthr)
            rst_out = out.detach()

            new_syn = torch.abs(alpha)*syn + input_activity[:, t]
            new_mem = (torch.abs(beta)*mem + syn)*(1.0-rst_out)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        # Now we merge the recorded membrane potentials into a single as_tensor
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        return spk_rec, mem_rec


class recurrent_layer:
    '''
    class to initialize and compute spiking recurrent layer
    '''
    def create_layer(nb_inputs, nb_outputs, fwd_scale, rec_scale):
        ff_layer = torch.empty(
            (nb_inputs, nb_outputs),  device=device, dtype=torch.float, requires_grad=True)
        torch.nn.init.normal_(ff_layer, mean=0.0,
                              std=fwd_scale/np.sqrt(nb_inputs))

        rec_layer = torch.empty(
            (nb_outputs, nb_outputs),  device=device, dtype=torch.float, requires_grad=True)
        torch.nn.init.normal_(rec_layer, mean=0.0,
                              std=rec_scale/np.sqrt(nb_inputs))
        return ff_layer,  rec_layer

    def compute_activity(nb_input, nb_neurons, input_activity, layer, nb_steps):
        syn = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=torch.float)
        mem = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=torch.float)
        out = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=torch.float)
        mem_rec = []
        spk_rec = []

        # Compute recurrent layer activity
        for t in range(nb_steps):
            # input activity plus last step output activity
            h1 = input_activity[:, t] + \
                torch.einsum("ab,bc->ac", (out, layer))
            mthr = mem-1.0
            out = spike_fn(mthr)
            rst = out.detach()  # We do not want to backprop through the reset

            new_syn = alpha*syn + h1
            new_mem = (beta*mem + syn)*(1.0-rst)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        # Now we merge the recorded membrane potentials into a single as_tensor
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        return spk_rec, mem_rec

    def compute_activity_tc(nb_input, nb_neurons, input_activity, layer, alpha, beta, nb_steps):
        syn = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=torch.float)
        mem = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=torch.float)
        out = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=torch.float)
        mem_rec = []
        spk_rec = []

        # Compute recurrent layer activity
        for t in range(nb_steps):
            # input activity plus last step output activity
            h1 = input_activity[:, t] + \
                torch.einsum("ab,bc->ac", (out, layer))
            mthr = mem-1.0
            out = spike_fn(mthr)
            rst = out.detach()  # We do not want to backprop through the reset

            new_syn = torch.abs(alpha)*syn + h1
            new_mem = (torch.abs(beta)*mem + syn)*(1.0-rst)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        # Now we merge the recorded membrane potentials into a single as_tensor
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        return spk_rec, mem_rec


class trainable_time_constants:
    def create_time_constants(nb_neurons, alpha_mean, beta_mean, trainable):
        alpha = torch.empty((nb_neurons),  device=device,
                            dtype=torch.float, requires_grad=trainable)
        torch.nn.init.normal_(
            alpha, mean=alpha_mean, std=alpha_mean/10)

        beta = torch.empty((nb_neurons),  device=device,
                           dtype=torch.float, requires_grad=trainable)
        torch.nn.init.normal_(
            beta, mean=beta_mean, std=beta_mean/10)
        return alpha, beta

class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 10

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input as_tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a as_tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors  # saved_as_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad

spike_fn = SurrGradSpike.apply


def run_NNI(args, params, name, ds_train, ds_test, ds_val):

    # Set the number of epochs
    eps = params["epochs"]

    #execution_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    #logging.info("Data storage initialized.\n")
    LOG.debug(print("{} data used.\n".format(name)))


    # Settings for the SNN
    global use_trainable_out
    use_trainable_out = False
    global use_trainable_tc
    use_trainable_tc = False
    global use_dropout
    use_dropout = False
    global batch_size
    batch_size = params["batch_size"] # 128
    global lr
    lr = params["lr"]

    """
    ### "debug" print:
    #logging.info("Setting up data.")
    ###
    # create train-test-validation split
    ratios = [70, 10, 20]

    # infile = open("./data_encoding", 'rb')
    infile = open(data_filepath, "rb")
    encoded_data = pickle.load(infile)
    infile.close()

    if original:
        encoded_label = input_currents.keys()
    else:
        # infile = open("./label_encoding", 'rb')
        infile = open(label_filepath, "rb")
        encoded_label = pickle.load(infile)
        infile.close()

    x_train, y_train, x_test, y_test, x_validation, y_validation = train_test_validation_split(
        np.array(encoded_data)[:, 0], encoded_label, split=ratios)

    labels_mapping = {
        'A': "Tonic spiking",
        'B': "Class 1",
        'C': "Spike frequency adaptation",
        'D': "Phasic spiking",
        'E': "Accommodation",
        'F': "Threshold variability",
        'G': "Rebound spike",
        'H': "Class 2",
        'I': "Integrator",
        'J': "Input bistability",
        'K': "Hyperpolarizing spiking",
        'L': "Hyperpolarizing bursting",
        'M': "Tonic bursting",
        'N': "Phasic bursting",
        'O': "Rebound burst",
        'P': "Mixed mode",
        'Q': "Afterpotentials",
        'R': "Basal bistability",
        'S': "Preferred frequency",
        'T': "Spike latency",
    }

    if ratios[2] > 0:
        data_steps = np.min(np.concatenate(([len(x) for x in x_train], [
                            len(x) for x in x_validation], [len(x) for x in x_test])), axis=0)
        x_train = torch.as_tensor(np.array(x_train), dtype=torch.float)
        labels_train = torch.as_tensor(value2index(
            y_train, labels_mapping), dtype=torch.long)
        x_test = torch.as_tensor(np.array(x_test), dtype=torch.float)
        labels_test = torch.as_tensor(value2index(
            y_test, labels_mapping), dtype=torch.long)
        x_validation = torch.as_tensor(
            np.array(x_validation), dtype=torch.float)
        labels_validation = torch.as_tensor(value2index(
            y_validation, labels_mapping), dtype=torch.long)
        ds_train = TensorDataset(x_train, labels_train)
        ds_test = TensorDataset(x_test, labels_test)
        ds_val = TensorDataset(x_validation, labels_validation)
    else:
        data_steps = np.min(np.concatenate(
            ([len(x) for x in x_train], [len(x) for x in x_test])), axis=0)
        x_train = torch.as_tensor(x_train, dtype=torch.float)
        labels_train = torch.as_tensor(value2index(
            y_train, labels_mapping), dtype=torch.long)
        x_test = torch.as_tensor(x_test, dtype=torch.float)
        labels_test = torch.as_tensor(value2index(
            y_test, labels_mapping), dtype=torch.long)
        ds_train = TensorDataset(x_train, labels_train)
        ds_test = TensorDataset(x_test, labels_test)
        ds_val = []
    """

    # tain the network (with validation)
    loss_hist, acc_hist, best_layers = nni_build_and_train(params, ds_train, ds_val, epochs=eps)

    # test the network (on never seen data) with weights from best validation
    test_acc, _ = compute_classification_accuracy(ds_test, best_layers)
    LOG.debug(print("Test accuracy for {}: {}%\n".format(
        name, np.round(test_acc*100, 2))))

    nni.report_final_result({"default": np.round(np.max(acc_hist[1])*100,4), # the default value is the maximum validation accuracy achieved
                             "Best training": np.round(np.max(acc_hist[0])*100,4),
                             "test": np.round(test_acc*100, 4)})
    
    return test_acc, best_layers


parser = argparse.ArgumentParser()

# Name (path) of the search space file (useful to auto-update the searchspace during the experiment)
parser.add_argument('-filename',
                    type=str,
                    default=searchspace_path,
                    help='Name (path) of the file for the search space.')
parser.add_argument('-epochs',
                    type=int,
                    default=300,
                    help='Number of training epochs.')
parser.add_argument('-batch_size',
                    type=int,
                    default=64,
                    help='Batch size.')
parser.add_argument('-lr',
                    type=float,
                    default=0.0001,
                    help='Learning rate.')
parser.add_argument('-nb_hidden',
                    type=int,
                    default=450,
                    help='Number of hidden neurons')
parser.add_argument('-fwd_weights_std',
                    type=float,
                    default=3,
                    help='The initial forward weights are drawn from a Gaussian '
                         'with a mean of 0 and astandard deviation of '
                         'init_weight_std * (1 - tau_syn) / sqrt(N), '
                         'where N is the number of neurons in the '
                         'previous layer and tau_syn the synaptic time '
                         'constant.')
parser.add_argument('-rec_weights_std',
                    type=float,
                    default=1,
                    help='The initial recurrent weights are drawn from a Gaussian '
                         'with a mean of 0 and astandard deviation of '
                         'init_weight_std * (1 - tau_syn) / sqrt(N), '
                         'where N is the number of neurons in the '
                         'previous layer and tau_syn the synaptic time '
                         'constant.')
parser.add_argument('-tau_mem',
                    type=float,
                    default=20e-3,
                    help='Membrane time constant (in s).')
parser.add_argument('-tau_syn',
                    type=float,
                    default=10e-3,
                    help='Synaptic time constant (in s).')
#parser.add_argument('-scale',
#                    type=float,
#                    default=10,
#                    help='Controls steepness of surrogate gradient.')
# ID of the running NNI experiment (useful to auto-update the searchspace during the experiment)
parser.add_argument('--id',
                    type=str,
                    default=nni.get_experiment_id(),
                    help="Experiment ID")

args = parser.parse_args()

params = vars(args)


try:

    # Specify what kind of data to use
    original = False
    fixed_length = not original
    noise = True
    jitter = True

    save_weights = True # to save weights from the best_layers variable

    # prepare data selection
    name = ""
    data_features = [original, fixed_length, noise, jitter]
    data_attributes = ["original", "fix_len", "noisy", "temp_jitter"]
    for num,el in enumerate(list(np.where(np.array(data_features)==True)[0])):
        name += "{} ".format(data_attributes[el])
    name = name[:-1]
    name = name.replace(" ","_")

    # load the test subset (always the same)
    ds_test = torch.load("./dataset_splits/{}/{}_ds_test.pt".format(name,name), map_location=device)
    
    # select random train and validation set.
    # To not mix train and validation data ALWAYS use same ID
    rnd_idx = np.random.randint(0, 10)  # take n_splits as max
    LOG.debug(print("\nSplit number {} (randomly) selected for this trial.".format(rnd_idx)))
    ds_train = torch.load("./dataset_splits/{}/{}_ds_train_{}.pt".format(name,name,rnd_idx), map_location=device)
    ds_val = torch.load("./dataset_splits/{}/{}_ds_val_{}.pt".format(name,name,rnd_idx), map_location=device)

    trial_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    LOG.debug(print("\nTrial {} (No. {}) started on: {}-{}-{} {}:{}:{}\n".format(nni.get_sequence_id()+1,
                                                                                 nni.get_sequence_id(),
                                                                                 trial_datetime[:4],
                                                                                 trial_datetime[4:6],
                                                                                 trial_datetime[6:8],
                                                                                 trial_datetime[-6:-4],
                                                                                 trial_datetime[-4:-2],
                                                                                 trial_datetime[-2:])))
    
    ### every n_tr, "update" the searchspace inducing a new RandomState for the tuner
    n_tr = 200
    if (nni.get_sequence_id() > 0) & (nni.get_sequence_id()%n_tr == 0):
        updater.update_searchspace(args) # it will use args.filename to update the search space

    # get parameters from the tuner combining them with the line arguments
    params_nni = nni.get_next_parameter()
    for ii in params_nni.keys():
        if ii in params.keys():
            del params[ii]

    PARAMS = {**params, **params_nni}
    #PARAMS = params  # when running out of NNI

    LOG.debug(PARAMS)
    LOG.debug(print("Parameters selected for trial {} (No. {}): {}\n".format(
        nni.get_sequence_id()+1, nni.get_sequence_id(), PARAMS)))

    # run network
    test_acc, best_layers = run_NNI(args, PARAMS, name, ds_train, ds_val, ds_test)

    # report results (i.e. test accuracy from best validation) of each trial
    path = './results/reports/{}'.format(name)
    create_directory(path)
    report_path = path + "/{}".format(nni.get_experiment_id())
    with open(report_path, 'a') as f:
        f.write("{} {} test accuracy (%)".format(str(test_acc*100),nni.get_trial_id()))
        f.write('\n')
    
    # save trained weights giving the highest test accuracy
    if save_weights:
        path = './results/layers/{}'.format(name)
        create_directory(path)
        save_layers_path = path + "/{}.pt".format(nni.get_experiment_id())
        with open(report_path, 'r') as f:
            if test_acc*100 >= np.max(np.asarray([(line.strip().split(" ")[0]) for line in f], dtype=np.float64)):
                torch.save(best_layers, save_layers_path)

except Exception as e:
    LOG.exception(e)
    raise
