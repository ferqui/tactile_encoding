"""
This script allows to train and test a network for
spiking activity classification with the parameters 
found from NNI optimization.
Such parameters are also saved to be re-used
independently of the NNI results database.

Settings to be accounted for:
    experiment_name
    do_training
    training_statistics
    nb_epochs
    experiment_id
    best_test_id
    save_weights
    save_fig
    store_weights
    trained_layers_path
    gpu_mem_frac
    use_seed

Fra, Vittorio,
Politecnico di Torino,
EDA Group,
Torino, Italy.
"""


import logging
import argparse
import numpy as np
import pandas as pd
import json
import random

import os
import datetime

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from NNI.utils.utils import set_device, gpu_usage_df, check_gpu_memory_constraint, create_directory, retrieve_nni_results, load_layers


### 1) various experiment settings #############################################

parser = argparse.ArgumentParser()

# Experiment name
parser.add_argument('-experiment_name',
                    type=str,
                    default="spike_classifier",
                    help='Name of this experiment.')
# Training needed or not
parser.add_argument('-do_training',
                    type=bool,
                    default=True,
                    help='If set to False, test only will be performed.')
# Make some statistics for training
parser.add_argument('-training_statistics',
                    type=bool,
                    default=True,
                    help='If set to True, multiple (5, by default) trainings will be performed (with use_seed consequently set to False).')
# Number of epochs
parser.add_argument('-nb_epochs',
                    type=int,
                    default=100,
                    help='Number of training epochs.')
# ID of the NNI experiment to refer to
parser.add_argument('-experiment_id',
                    type=str,
                    default="vpeqjlkr",
                    help='ID of the NNI experiment whose results are to be used.')
# ID of the NNI trial providing the best test accuracy
parser.add_argument('-best_test_id',
                    type=str,
                    default="euX7c",
                    help='ID of the NNI trial that gave the highest test accuracy.')
# Save the weights (to be re-used right after the training to test) or not
parser.add_argument('-save_weights',
                    type=bool,
                    default=True,
                    help='Weights can be saved to be loaded after training and used for test.')
# Save figures
parser.add_argument('-save_fig',
                    type=bool,
                    default=True,
                    help='Save or not the plots produced during training and test.')
# Store the weights 
parser.add_argument('-store_weights',
                    type=bool,
                    default=True,
                    help='Weights can be stored with specific, unique name.')
# Path of weights to perform test only (if do_training is False)
parser.add_argument('-trained_layers_path',
                    type=str,
                    default="./NNI/results/layers/fix_len_noisy_temp_jitter/vpeqjlkr_backup.pt",
                    help='Path of the weights to be loaded to perform test only (given do_training is set to False).')
# (maximum) GPU memory fraction to be allocated
parser.add_argument('-gpu_mem_frac',
                    type=float,
                    default=0.3,
                    help='The maximum GPU memory fraction to be used by this experiment.')
# Set seed usage
parser.add_argument('-use_seed',
                    type=bool,
                    default=True,
                    help='Set if a seed is to be used or not.')

args = parser.parse_args()

settings = vars(args)

experiment_name = settings["experiment_name"]

do_training = settings["do_training"]
training_statistics = settings["training_statistics"]

experiment_id = settings["experiment_id"]
if do_training:
    best_test_id, _ = retrieve_nni_results(experiment_name, experiment_id, "test")
else:
    trained_layers_path = settings["trained_layers_path"]
    best_test_id = settings["best_test_id"]

save_weights = settings["save_weights"]
save_fig = settings["save_fig"]
store_weights = settings["store_weights"]

nb_epochs = settings["nb_epochs"]

use_seed = settings["use_seed"] # it will be in any case "re-set" to False for test statistics

if not training_statistics:
    if use_seed:
        seed = 42
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    else:
        seed = None
else:
    use_seed = False
    seed = None

################################################################################


experiment_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


### 2) data "configuration" specific for spike classification from MN paper ####
"""
Data created following the paper "A Generalized
Linear Integrate-and-Fire Neural Model Produces Diverse Spiking 
Behaviors" by Stefan Mihalas and Ernst Niebur.

Muller-Cleve, Simon F.,
Istituto Italiano di Tecnologia - IIT,
Event-driven perception in robotics - EDPR,
Genova, Italy.
"""
 
# Specify what kind of data to use
original = False
fixed_length = not original
noise = True
jitter = True

# Prepare data selection
name = ""
data_features = [original, fixed_length, noise, jitter]
data_attributes = ["original", "fix_len", "noisy", "temp_jitter"]
for num,el in enumerate(list(np.where(np.array(data_features)==True)[0])):
    name += "{} ".format(data_attributes[el])
name = name[:-1]
name = name.replace(" ","_")

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

################################################################################


### 3) log file configuration ##################################################

log_path = "./logs/optimized/{}/{}".format(experiment_name,name)
create_directory(log_path)
logging.basicConfig(filename=log_path+"/{}_{}.log".format(experiment_id,best_test_id),
                    filemode='a',
                    format="%(asctime)s %(name)s %(message)s",
                    datefmt='%Y%m%d_%H%M%S')
LOG = logging.getLogger(experiment_name)
LOG.setLevel(logging.DEBUG)
LOG.debug("Experiment started on: {}-{}-{} {}:{}:{}\n".format(
    experiment_datetime[:4],
    experiment_datetime[4:6],
    experiment_datetime[6:8],
    experiment_datetime[-6:-4],
    experiment_datetime[-4:-2],
    experiment_datetime[-2:])
    )

if use_seed:
    LOG.debug("Seed set to {}\n".format(seed))

################################################################################


### 4) CUDA device set-up ######################################################

gpu_mem_frac = settings["gpu_mem_frac"]
flag_allocate_memory = False
flag_print = True
while not flag_allocate_memory:
    if check_gpu_memory_constraint(gpu_usage_df(),gpu_mem_frac):
        flag_allocate_memory = True
        print("The available memory is enough.")
    else:
        if flag_print:
            print("Waiting for more memory available.")
            flag_print = False
device = set_device(auto_sel=True, gpu_mem_frac=gpu_mem_frac)

################################################################################


### 5) data and parameters paths to be used ####################################

# Load the test subset (always the same)
ds_test = torch.load("./dataset_splits/{}/{}_ds_test.pt".format(name,name), map_location=device)

nb_steps = len(next(iter(ds_test))[0])

if do_training:

    if not training_statistics:
        # Select random training and validation set
        rnd_idx = np.random.randint(0, 10) # 3
        LOG.debug("Split number {} used for this experiment.\n".format(rnd_idx))
        ds_train = torch.load("./dataset_splits/{}/{}_ds_train_{}.pt".format(name,name,rnd_idx), map_location=device)
        ds_val = torch.load("./dataset_splits/{}/{}_ds_val_{}.pt".format(name,name,rnd_idx), map_location=device)

    # Get the optimized parameters
    parameters_path = './NNI/results/parameters/best_test/{}/{}/{}.json'.format(experiment_name,name,experiment_id)
    with open(parameters_path, 'r') as fp:
        params = json.load(fp)

    # Store the optimized parameters
    parameters_path = './parameters/optimized/{}/{}'.format(experiment_name,name)
    create_directory(parameters_path)
    with open(parameters_path+"/parameters.json", 'w') as fp:
        json.dump(params, fp)

else:

    parameters_path = './parameters/optimized/{}/{}'.format(experiment_name,name)
    with open(parameters_path, 'r') as fp:
        params = json.load(fp)

################################################################################


### 6)  temporal dynamics quantities for the SNN ###############################

tau_mem = params["tau_mem"]
tau_syn = params["tau_syn"]
dt = 1e-3
alpha = torch.as_tensor(float(np.exp(-dt/tau_syn)))
beta = torch.as_tensor(float(np.exp(-dt/tau_mem)))

################################################################################



### Various definitions ########################################################

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


def train_validate_test(params, name, ds_train, ds_val, ds_test):

    # Set the number of epochs
    eps = nb_epochs

    LOG.debug("{} data used.\n".format(name))

    # Train the network with validation
    loss_hist, acc_hist, best_layers = build_and_train(params, ds_train, ds_val, epochs=eps)
    LOG.debug("Best validation accuracy: {}%".format(np.round(np.max(acc_hist[1])*100,4)))
    LOG.debug("Best training accuracy: {}%".format(np.round(np.max(acc_hist[0])*100,4)))

    # Test the network (on never seen data) with weights from best validation
    test_acc, _ = compute_classification_accuracy(params, ds_test, best_layers)
    LOG.debug("Test accuracy from best validation: {}%\n".format(np.round(test_acc*100, 4)))
    
    return loss_hist, acc_hist, test_acc, best_layers


def build_and_train(params, ds_train, ds_val, epochs=300):

    # Network parameters
    nb_inputs = 1
    nb_outputs = 20  # number of spiking behaviours from MN paper
    nb_hidden = int(params["nb_hidden"])
    fwd_weight_scale = params["fwd_weights_std"]
    rec_weight_scale = params["rec_weights_std"]

    lr = params["lr"]

    # Spiking network
    layers = []
    # recurrent layer
    w1, v1 = recurrent_layer.create_layer(
        nb_inputs, nb_hidden, fwd_weight_scale, rec_weight_scale)
    # readout layer
    w2 = feedforward_layer.create_layer(
        nb_hidden, nb_outputs, fwd_weight_scale)
    layers.append(w1), layers.append(w2), layers.append(v1)

    layers_init = []
    for ii in layers:
        layers_init.append(ii.detach().clone())

    opt_parameters = [w1, w2, v1]

    loss_hist, accs_hist, best_layers = train_net(
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

    LOG.debug("------------------------------------------------------------------------------------")
    LOG.debug("Final results: ")
    LOG.debug("Best training accuracy: {:.2f}% and according validation accuracy: {:.2f}% at epoch: {}".format(
        acc_best_train, acc_val_at_best_train, idx_best_train+1))
    LOG.debug("Best validation accuracy: {:.2f}% and according training accuracy: {:.2f}% at epoch: {}".format(
        acc_best_val, acc_train_at_best_val, idx_best_val+1))
    LOG.debug("------------------------------------------------------------------------------------\n")
    
    return loss_hist, accs_hist, best_layers


def train_net(
    dataset,
    lr=0.0015,
    nb_epochs=300,
    opt_parameters=None,
    layers=None,
    dataset_val=None
    ):

    if (opt_parameters != None) & (layers != None):
        parameters = opt_parameters  # The paramters we want to optimize
        layers = layers
    elif (opt_parameters != None) & (layers == None):
        layers = [w1, w2, v1]
    elif (opt_parameters == None) & (layers != None):
        layers = layers
    elif (opt_parameters == None) & (layers == None):
        parameters = [w1, w2, v1]
        layers = [w1, w2, v1]

    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

    batch_size = params["batch_size"] # 128

    generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # The optimization loop
    loss_hist = [[], []]
    accs_hist = [[], []]
    
    for e in range(nb_epochs):
        
        optimizer = torch.optim.Adamax(
            parameters, lr=lr, betas=(0.9, 0.995))
        
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
            m = torch.sum(spks_out, 1)  # sum over time

            # cross entropy loss on the active read-out layer
            log_p_y = log_softmax_fn(m)

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
                params,
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

        LOG.debug("Epoch {}/{} \t --> \ttraining accuracy (loss): {:.2f}% ({:.5f}), \tvalidation accuracy (loss): {:.2f}% ({:.5f})".format(
            e + 1, nb_epochs, accs_hist[0][-1]*100, loss_hist[0][-1], accs_hist[1][-1]*100, loss_hist[1][-1]))
        
        if (e+1)%10 == 0:
            print("\tepoch {}/{} done ({}) \t --> \ttraining accuracy (loss): {:.2f}% ({:.5f}), \tvalidation accuracy (loss): {:.2f}% ({:.5f})".format(e+1,nb_epochs,datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),accs_hist[0][-1]*100, loss_hist[0][-1], accs_hist[1][-1]*100, loss_hist[1][-1]))

    return loss_hist, accs_hist, best_acc_layers


def run_snn(
    inputs,
    layers,
    ):

    w1, w2, v1 = layers

    # Network parameters
    nb_input_copies = 1
    nb_outputs = 20  # number of spiking behaviours from MN paper
    nb_hidden = int(params["nb_hidden"])

    bs = inputs.shape[0]

    h1 = torch.einsum(
        "abc,cd->abd", (inputs.tile((nb_input_copies,)), w1))
    spk_rec, mem_rec = recurrent_layer.compute_activity(
        bs, nb_hidden, h1, v1, nb_steps)

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))
    s_out_rec, out_rec = feedforward_layer.compute_activity(
        bs, nb_outputs, h2, nb_steps)

    other_recs = [mem_rec, spk_rec, out_rec]
    layers_update = layers

    return s_out_rec, other_recs, layers_update


def build_and_test(
    params,
    ds_test,
    trained_path,
    device=device,
    N=10,
    ):
    
    # Load the pre-trained weights
    layers = load_layers(trained_path, map_location=device, requires_grad=False)

    test_N = []

    for ii in range(N):

        test_acc, _, _ = compute_classification_accuracy(params, ds_test, layers=layers, label_probabilities=True, shuffle=True, use_seed=False)
        
        test_N.append(test_acc)
        LOG.debug("Test {}/{}: {}%".format(ii+1,N,np.round(test_acc*100,4)))

    LOG.debug("Min. test accuracy: {}%".format(np.round(np.min(test_N)*100,4)))
    LOG.debug("Max. test accuracy: {}%".format(np.round(np.max(test_N)*100,4)))
    LOG.debug("Mean test accuracy: {}%".format(np.round(np.mean(test_N)*100,4)))
    LOG.debug("Median test accuracy: {}%".format(np.round(np.median(test_N)*100,4)))
    LOG.debug("Std. Dev. test accuracy: {}%\n".format(np.round(np.std(test_N)*100,4)))
    
    # N single-sample inferences to check label probbailities
    for ii in range(N):
        single_sample = next(iter(DataLoader(ds_test, batch_size=1, shuffle=True, num_workers=0)))
        _, _, lbl_probs = compute_classification_accuracy(params, TensorDataset(single_sample[0],single_sample[1]), layers, label_probabilities=True)
        LOG.debug("Single-sample inference {}/{} from test set:".format(ii+1,10))
        LOG.debug("Sample: {} \tPrediction: {}".format(list(labels_mapping.keys())[single_sample[1]],list(labels_mapping.keys())[torch.max(lbl_probs.cpu(),1)[1]]))
        LOG.debug("Label probabilities (%): {}".format(np.round(np.array(lbl_probs.detach().cpu().numpy())*100,2)))

    LOG.debug("---------------------------------------------------------------------------------------------------\n\n")
    
    ConfusionMatrix(params, ds_test, save_fig, layers=layers, labels=list(labels_mapping.keys()), use_seed=False)


def compute_classification_accuracy(params, dataset, layers=None, label_probabilities=False, shuffle=False, use_seed=use_seed):
    """ Computes classification accuracy on supplied data in batches. """

    if use_seed:
        g = torch.Generator()
        g.manual_seed(seed)
        generator = DataLoader(dataset, batch_size=params["batch_size"], shuffle=shuffle, num_workers=0, generator=g)
    else:
        generator = DataLoader(dataset, batch_size=params["batch_size"], shuffle=shuffle, num_workers=0)

    accs = []
    losss = []
    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        if layers == None:
            layers = [w1, w2, v1]
            spks_out, _, _ = run_snn(inputs=x_local, layers=layers)
        else:
            spks_out, _, _ = run_snn(inputs=x_local, layers=layers)
        # with output spikes
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


def ConfusionMatrix(params, dataset, save, title=False, layers=None, labels=None, use_seed=use_seed):
        
    if use_seed:
        g = torch.Generator()
        g.manual_seed(seed)
        generator = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True,
                            num_workers=0, generator=g)
    else:
        generator = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True, num_workers=0)
        
    accs = []
    trues = []
    preds = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(
            device, non_blocking=True), y_local.to(device, non_blocking=True)
        if layers == None:
            layers = [w1, w2, v1]
            spks_out, _, _ = run_snn(inputs=x_local, layers=layers)
        else:
            spks_out, _, _ = run_snn(inputs=x_local, layers=layers)
        # with output spikes
        m = torch.sum(spks_out, 1)  # sum over time
        _, am = torch.max(m, 1)     # argmax over output units
        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
        trues.extend(y_local.detach().cpu().numpy())
        preds.extend(am.detach().cpu().numpy())

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
    if title:
        plt.title("Accuracy from confusion matrix: {:.2f}% +- {:.2f}%\n".format(np.median(accs) * 100, np.std(accs)*100))
    plt.xlabel('\nPredicted')
    plt.ylabel('True\n')
    plt.xticks(rotation=0)
    if save:
        path_for_plots = "./results/plots/optimized/{}/{}".format(experiment_name,name)
        create_directory(path_for_plots)
        #path_to_save_fig = f'{path_for_plots}/generation_{generation+1}_individual_{best_individual+1}'
        path_to_save_fig = f'{path_for_plots}/cm_{experiment_id}_{best_test_id}'
        #path_to_save_fig = f'{path_to_save_fig}_cm.png'
        path_to_save_fig = f'{path_to_save_fig}_{experiment_datetime}'
        plt.savefig(path_to_save_fig+".png", dpi=300)
        plt.savefig(path_to_save_fig+".pdf", dpi=300)
        plt.close()
    else:
        plt.show()

################################################################################



### WHERE THINGS ACTUALLY HAPPEN ###############################################

print("EXPERIMENT STARTED --- {}-{}-{} {}:{}:{}".format(
    experiment_datetime[:4],
    experiment_datetime[4:6],
    experiment_datetime[6:8],
    experiment_datetime[-6:-4],
    experiment_datetime[-4:-2],
    experiment_datetime[-2:])
    )

if do_training:

    # Path for plots from training and validation
    if save_fig:
        path_for_plots = "./results/plots/optimized/{}/{}".format(experiment_name,name)
        create_directory(path_for_plots)

    if training_statistics:
        """
        Muller-Cleve, Simon F.,
        Istituto Italiano di Tecnologia - IIT,
        Event-driven perception in robotics - EDPR,
        Genova, Italy.
        """
        
        repetitions = 5

        loss_train_list = []
        acc_train_list = []
        loss_val_list = []
        acc_val_list = []
        acc_test_list = []

        print("*** training (with validation) statistics started ***".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        LOG.debug("### Training statistics with {} repetitions started ({}). ###\n".format(repetitions,datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        
        for rpt in range(repetitions):
            # Reload data for each repetition
            # Select random training and validation set
            rnd_idx = np.random.randint(0, 10) # 3
            LOG.debug("Repetition {}/{}: started ({}) with split number {}.\n".format(rpt+1,repetitions,datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),rnd_idx))
            ds_train = torch.load("./dataset_splits/{}/{}_ds_train_{}.pt".format(name,name,rnd_idx), map_location=device)
            ds_val = torch.load("./dataset_splits/{}/{}_ds_val_{}.pt".format(name,name,rnd_idx), map_location=device)

            # Train the network with validation and test
            loss_hist, acc_hist, test_acc, best_layers = train_validate_test(params, name, ds_train, ds_val, ds_test)

            # Save layers providing the best test accuracy
            if rpt == 0:
                very_best_layer = best_layers
                best_acc = test_acc
            else:
                if test_acc > best_acc:
                    very_best_layer = best_layers
                    best_acc = test_acc

            loss_train_list.append(loss_hist[0])
            acc_train_list.append(acc_hist[0])
            loss_val_list.append(loss_hist[1])
            acc_val_list.append(acc_hist[1])
            acc_test_list.append(test_acc)

            print("\trepetition {}/{} done ({}) --> test accuracy: {}%".format(rpt+1,repetitions,datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),np.round(test_acc*100,4)))
        
        best_layers = very_best_layer

        LOG.debug("Overall best training accuracy: {}%".format(np.round(np.nanmax(acc_train_list)*100,4)))
        LOG.debug("Overall best validation accuracy: {}%".format(np.round(np.nanmax(acc_val_list)*100,4)))
        LOG.debug("Overall best test accuracy: {}%\n".format(np.round(best_acc*100,4)))

        # Make plots for loss and accuracy from training and validation
        # Accuracy:
        # Compute mean, median and std. dev.
        acc_mean_train = np.mean(acc_train_list, axis=0)
        acc_median_train = np.median(acc_train_list, axis=0)
        acc_std_train = np.std(acc_train_list, axis=0)
        acc_mean_val = np.mean(acc_val_list, axis=0)
        acc_median_val = np.median(acc_val_list, axis=0)
        acc_std_val = np.std(acc_val_list, axis=0)
        ## Identify repetition with the best validation accuracy
        #best_rpt, best_val_idx = np.where(np.max(acc_val_list) == acc_val_list)
        #best_rpt, best_val_idx = best_rpt[0], best_val_idx[0]
        plt.figure()
        ## Plot the identified repetition
        #plt.plot(range(1, len(acc_train_list[best_rpt])+1), 100*np.array(
        #    acc_train_list[best_rpt]), color='blue', linestyle='dashed')
        #plt.plot(range(1, len(acc_val_list[best_rpt])+1), 100*np.array(
        #    acc_val_list[best_rpt]), color='orangered', linestyle='dashed')
        # Plot the "median repetition" of training and validation
        plt.plot(range(1, len(acc_median_train)+1),
                 100*np.array(acc_median_train), color='blue')
        plt.plot(range(1, len(acc_median_val)+1), 100 *
                 np.array(acc_median_val), color='orangered')
        plt.fill_between(range(1, len(acc_median_train)+1), 100*(acc_median_train+acc_std_train), 100*(
            acc_median_train-acc_std_train), color='cornflowerblue')
        plt.fill_between(range(1, len(acc_median_val)+1), 100*(
            acc_median_val+acc_std_val), 100*(acc_median_val-acc_std_val), color='sandybrown')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.ylim((0, 105))
        plt.legend(["Training", "Validation"], loc='lower right')
        plt.show()
        if save_fig:
            plt.savefig(path_for_plots + "/accuracy_{}_{}_{}_stats.pdf".format(experiment_id,best_test_id,experiment_datetime), dpi=300)
            plt.savefig(path_for_plots + "/accuracy_{}_{}_{}_stats.png".format(experiment_id,best_test_id,experiment_datetime), dpi=300)
        # Loss:
        # Compute mean, median and std. dev.
        loss_mean_train = np.mean(loss_train_list, axis=0)
        loss_median_train = np.median(loss_train_list, axis=0)
        loss_std_train = np.std(loss_train_list, axis=0)
        loss_mean_val = np.mean(loss_val_list, axis=0)
        loss_median_val = np.median(loss_val_list, axis=0)
        loss_std_val = np.std(loss_val_list, axis=0)
        plt.figure()
        # Plot the "median repetition" of training and validation
        plt.plot(range(1, len(loss_median_train)+1), np.array(loss_median_train), color='tab:red')
        plt.plot(range(1, len(loss_median_val)+1), np.array(loss_median_val), color='tab:green')
        plt.fill_between(range(1, len(loss_median_train)+1), loss_median_train+loss_std_train, loss_median_train-loss_std_train, color='lightcoral')
        plt.fill_between(range(1, len(loss_median_val)+1), loss_median_val+loss_std_val, loss_median_val-loss_std_val, color='lightgreen')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(bottom=0)
        plt.legend(["Training", "Validation"], loc='lower right')
        plt.show()
        if save_fig:
            plt.savefig(path_for_plots + "/loss_{}_{}_{}_stats.pdf".format(experiment_id,best_test_id,experiment_datetime), dpi=300)
            plt.savefig(path_for_plots + "/loss_{}_{}_{}_stats.png".format(experiment_id,best_test_id,experiment_datetime), dpi=300)

        LOG.debug("### Training statistics done ({}). ###\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        print("*** training (with validation) statistics done ***")

    else:

        # Train the network with validation and test
        print("*** training with validation started ***")
        loss_hist, acc_hist, test_acc, best_layers = train_validate_test(params, name, ds_train, ds_val, ds_test)
        print("*** training with validation done ***")

        # Make plots from training and validation
        # Accuracy:
        plt.figure()
        plt.plot(range(1, len(acc_hist[0])+1), 100 *
                 np.array(acc_hist[0]), color='blue')
        plt.plot(range(1, len(acc_hist[1])+1), 100 *
                 np.array(acc_hist[1]), color='orangered')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.ylim((0, 105))
        #plt.title("{} ({} epochs)".format(name,nb_epochs))
        plt.legend(["Training", "Validation"], loc='lower right')
        if save_fig:
            plt.savefig(path_for_plots + "/accuracy_{}_{}_{}.pdf".format(experiment_id,best_test_id,experiment_datetime), dpi=300)
            plt.savefig(path_for_plots + "/accuracy_{}_{}_{}.png".format(experiment_id,best_test_id,experiment_datetime), dpi=300)
        plt.show()
        if save_fig:
            print("*** accuracy plot saved ***")
        # Loss:
        plt.figure()
        plt.plot(range(1, len(loss_hist[0])+1),
                 np.array(loss_hist[0]), color='tab:red')
        plt.plot(range(1, len(loss_hist[1])+1),
                 np.array(loss_hist[1]), color='tab:green')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(bottom=0)
        #plt.title("{} ({} epochs)".format(name,nb_epochs))
        plt.legend(["Training", "Validation"], loc='upper right')
        if save_fig:
            plt.savefig(path_for_plots + "/loss_{}_{}_{}.pdf".format(experiment_id,best_test_id,experiment_datetime), dpi=300)
            plt.savefig(path_for_plots + "/loss_{}_{}_{}.png".format(experiment_id,best_test_id,experiment_datetime), dpi=300)
        plt.show()
        if save_fig:
            print("*** loss plot saved ***")
        
    # Save (to re-load) trained weights 
    path = './results/layers/optimized/{}/{}'.format(experiment_name,name)
    create_directory(path)
    save_layers_path = path + "/{}.pt".format(experiment_id)
    torch.save(best_layers, save_layers_path)
    print("*** weights saved ***")

    # Save (to store) trained weights
    if store_weights:
        path = './results/layers/optimized/{}/{}'.format(experiment_name,name)
        create_directory(path)
        save_layers_path = path + "/{}_{}.pt".format(experiment_id,experiment_datetime)
        torch.save(best_layers, save_layers_path)
        print("*** weights stored ***")
        trained_layers_path = save_layers_path

# Test the network with statistics
print("*** test statistics started ***")
build_and_test(params, ds_test, trained_layers_path, N=50)
print("*** test statistics done ***")

conclusion_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print("EXPERIMENT DONE --- {}-{}-{} {}:{}:{}".format(
    conclusion_datetime[:4],
    conclusion_datetime[4:6],
    conclusion_datetime[6:8],
    conclusion_datetime[-6:-4],
    conclusion_datetime[-4:-2],
    conclusion_datetime[-2:])
    )

################################################################################
