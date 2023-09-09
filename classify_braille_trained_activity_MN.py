"""
##### NOTE: work in progress (last modified 2023.09.09 - 17:15)
#####
##### History:
#####   - added comments to identify key changes to be implemented
#####   - except for the input data name (since they must be replaced with Braille data) "original_input" replaced with "braille_trained"
#####   - trained MN parameters set as default
#####   - plotting part commented and save_fig set to False
#####   - GPU settings and selection updated
#####   - Braille data loading added
#####   - Dataframe to collect behaviour predictions created
#####   - dt fixed
#####   - "debugging mode" for log file added
#####   - nb_steps definition in classify_spikes modified

This script allows to classify MN neuron output 
spike patterns obtained from an NNI-optimized
network with trained MN neuron parameters for 
Braille letters classification.
The spike_classifier with NNI-optimized parameters
and pre-trained weights is used.

Settings to be accounted for:
    experiment_name
    n_samples
    threshold
    frequency
    experiment_id
    best_test_id
    trained_layers_path
    auto_gpu
    manual_gpu_idx
    gpu_mem_frac
    save_fig
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
from collections import namedtuple

import os
import datetime

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from NNI.utils.utils import set_device, gpu_usage_df, check_gpu_memory_constraint, create_directory, load_layers

from data.datasets import load_data


### 1) various experiment settings #############################################

parser = argparse.ArgumentParser()

# Experiment name
parser.add_argument('-experiment_name',
                    type=str,
                    default="braille_trained_activity_classifier",
                    help='Name of this experiment.')
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
# Path of weights to perform test only (if do_training is False)
parser.add_argument('-trained_layers_path',
                    type=str,
                    default="./results/layers/optimized/spike_classifier/fix_len_noisy_temp_jitter/vpeqjlkr_ref.pt",
                    help='Path of the weights to be loaded to perform test only (given do_training is set to False).')
# Auto-selection of GPU
parser.add_argument('-auto_gpu',
                    type=bool,
                    default=True,
                    help='Enable or not auto-selection of GPU to use.')
# Manual selection of GPU
parser.add_argument('-manual_gpu_idx',
                    type=int,
                    default=1,
                    help='Set which GPU to use.')
# (maximum) GPU memory fraction to be allocated
parser.add_argument('-gpu_mem_frac',
                    type=float,
                    default=0.3,
                    help='The maximum GPU memory fraction to be used by this experiment.')
# Save figures
parser.add_argument('-save_fig',
                    type=bool,
                    default=False,
                    help='Save or not the plots of the spiking patterns.')
# Set seed usage
parser.add_argument('-use_seed',
                    type=bool,
                    default=True,
                    help='Set if a seed is to be used or not.')
# Specify if running for debug
parser.add_argument('-debugging',
                    type=bool,
                    default=True,
                    help='Set if the run is to debug the code or not.')

args = parser.parse_args()

settings = vars(args)

experiment_name = settings["experiment_name"]

experiment_id = settings["experiment_id"]
trained_layers_path = settings["trained_layers_path"]
best_test_id = settings["best_test_id"]

save_fig = settings["save_fig"]

use_seed = settings["use_seed"]

if use_seed:
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
else:
    seed = None

################################################################################


experiment_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


### 2) data loading ############################################################

braille_data_path = "./data/100Hz/data_braille_letters_all.pkl"

#braille_data = np.array(pd.read_pickle(braille_data_path))

data, labels, _, _, _, _ = load_data(braille_data_path)

letter_written = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

letters = [letter_written[ii] for ii in labels]

# For the activity classification:
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

log_path = "./logs/classification/braille_trained_activity_MN"
create_directory(log_path)
if settings["debugging"]:
    logging.basicConfig(filename=log_path+"/{}_{}_debug.log".format(experiment_id,best_test_id),
                        filemode='a',
                        format="%(asctime)s %(name)s %(message)s",
                        datefmt='%Y%m%d_%H%M%S')
else:
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
if settings["auto_gpu"]:
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
else:
    gpu_sel = settings["manual_gpu_idx"]
    print("Single GPU manually selected. Setting up the simulation on {}".format("cuda:"+str(gpu_sel)))
    device = torch.device("cuda:"+str(gpu_sel))
    torch.cuda.set_per_process_memory_fraction(gpu_mem_frac, device=device)

################################################################################


### 5) NNI-optimized spike_classifier hyperparameters ##########################

parameters_path = './NNI/results/parameters/best_test/spike_classifier/fix_len_noisy_temp_jitter/{}.json'.format(experiment_id)
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


class MN_neuron_braille_trained(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'i1', 'i2', 'Thr', 'spk'])

    """
    # Trained MN parameters:
    {
        "A1": -0.015625353902578354, 
        "G": 45.24007797241211, 
        "a": 2.6239240169525146, 
        "A2": -1.0590057373046875, 
        "k2": 20.0, 
        "b": 12.77495288848877, 
        "R2": -1.1421641111373901, 
        "R1": 0.3858567178249359, 
        "k1": 200.0
    }
    """

    def __init__(self, nb_inputs, parameters_combination,
                 dt=1/100,
                 a=2.6239240169525146, 
                 A1=-0.015625353902578354, 
                 A2=-1.0590057373046875, 
                 b=12.77495288848877, 
                 G=45.24007797241211, 
                 k1=200.0, 
                 k2=20.0, 
                 R1=0.3858567178249359, 
                 R2=-1.1421641111373901,
                 C=1, 
                 train=False):  
        super(MN_neuron_braille_trained, self).__init__()

        # One-to-one synapse
        self.linear = nn.Parameter(torch.ones(1, nb_inputs), requires_grad=train)

        self.C = C

        self.N = nb_inputs

        self.EL = -0.07
        self.Vr = -0.07
        self.Tr = -0.06
        self.Tinf = -0.05

        self.a = a
        self.A1 = A1
        self.A2 = A2
        self.b = b  # units of 1/s
        self.G = G * self.C  # units of 1/s
        self.k1 = k1  # units of 1/s
        self.k2 = k2  # units of 1/s
        self.R1 = R1
        self.R2 = R2

        self.dt = dt # get dt from sample rate!

        if parameters_combination != None:
            parameters_list = ["a", "A1", "A2", "b", "G", "k1", "k2", "R1", "R2"]
            for ii in parameters_list:
                if ii in list(parameters_combination.keys()):
                    eval_string = "self.{}".format(ii) + " = " + str(parameters_combination[ii])
                    exec(eval_string)

        one2N_matrix = torch.ones(1, nb_inputs)

        self.a = nn.Parameter(one2N_matrix * self.a, requires_grad=train)
        
        self.A1 = nn.Parameter(one2N_matrix * self.A1 * self.C, requires_grad=train)
        self.A2 = nn.Parameter(one2N_matrix * self.A2 * self.C, requires_grad=train)

        self.state = None

    def forward(self, x):

        if self.state is None:
            self.state = self.NeuronState(V=torch.ones(x.shape[1], self.N, device=x.device) * self.EL,
                                          i1=torch.zeros(x.shape[1], self.N, device=x.device),
                                          i2=torch.zeros(x.shape[1], self.N, device=x.device),
                                          Thr=torch.ones(x.shape[1], self.N, device=x.device) * self.Tinf,
                                          spk=torch.zeros(x.shape[1], self.N, device=x.device))

        V = self.state.V
        i1 = self.state.i1
        i2 = self.state.i2
        Thr = self.state.Thr

        activity_spikes = []

        for t in range(x.shape[0]):

            i1 += -self.k1 * i1 * self.dt
            i2 += -self.k2 * i2 * self.dt
            V += self.dt * (self.linear * x[t] + i1 + i2 - self.G * (V - self.EL)) / self.C
            Thr += self.dt * (self.a * (V - self.EL) - self.b * (Thr - self.Tinf))

            spk = spike_fn(V - Thr)
            activity_spikes.append(spk.detach().flatten().cpu().numpy())

            i1 = (1 - spk) * i1 + (spk) * (self.R1 * i1 + self.A1)
            i2 = (1 - spk) * i2 + (spk) * (self.R2 * i2 + self.A2)
            Thr = (1 - spk) * Thr + (spk) * torch.max(Thr, torch.tensor(self.Tr))
            V = (1 - spk) * V + (spk) * self.Vr

            self.state = self.NeuronState(V=V, i1=i1, i2=i2, Thr=Thr, spk=spk)

        return np.array(activity_spikes) #spk

    def reset(self):
        self.state = None


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


def run_snn(
    inputs,
    nb_steps,
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


def classify_spikes(input_spikes, single_input, labels_mapping, trained_path, device=device):
    
    # Load the pre-trained weights
    layers = load_layers(trained_path, map_location=device, requires_grad=False)

    if single_input:
        if type(input_spikes) != torch.Tensor:
            input_spikes = torch.as_tensor(input_spikes, dtype=torch.float, device=device)
        single_sample = torch.reshape(input_spikes, (1,input_spikes.shape[0],input_spikes.shape[1])) # (batch_size, time, channels)
    else:
        rnd_idx = np.random.randint(0, input_spikes.shape[0])
        single_sample = torch.as_tensor(np.array(input_spikes[rnd_idx,:]), dtype=torch.float, device=device)
        single_sample = torch.reshape(single_sample, (1,single_sample.shape[0],1)) # (batch_size, time, channels)

    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)

    spks_out, _, _ = run_snn(inputs=single_sample, nb_steps=activity_spikes.shape[0], layers=layers)

    m = torch.sum(spks_out, 1)  # sum over time
    _, am = torch.max(m, 1)     # argmax over output units
    pred = list(labels_mapping.keys())[am] # MN-defined label of the spiking behaviour

    log_p_y = log_softmax_fn(m)

    if single_input:
        return pred, torch.exp(log_p_y) # i.e.: predicted label, labels probabilities
    else:
        return rnd_idx, pred, torch.exp(log_p_y) # i.e.: random sample, predicted label, labels probabilities

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

### Perform spiking patterns classification
# # The plotting part is comented and must be checked and adapted in case it's needed to produce some figure
# if save_fig:
#     path_for_plots = "./results/plots/classification/braille_trained_activity_MN"
#     create_directory(path_for_plots)
print("*** classification started ***")
activity_classification = pd.DataFrame()
letter = []
behaviour = []
behaviour_probs = []
for num,el in enumerate(data):
    LOG.debug("Single-sample inference {}/{} from Braille-trained activity of the Braille classifier:".format(num+1,len(data)))
    LOG.debug("Letter: {}\n".format(letters[num]))
    encoder_MN = MN_neuron_braille_trained(nb_inputs=1, parameters_combination=None).to(device)
    #nb_steps = el.shape[0]
    for ch in range(el.shape[1]):
        input_signal = torch.as_tensor(torch.reshape(el[:,ch],(len(el[:,ch]),1)), dtype=torch.float, device=device)
        activity_spikes = encoder_MN(input_signal)
        pred, probs = classify_spikes(activity_spikes, True, labels_mapping, trained_layers_path)
        letter.append(letters[num])
        behaviour.append(pred)
        behaviour_probs.append(np.round(np.array(probs.detach().cpu().numpy())*100,2))
        LOG.debug("Behaviour prediction (neuron {}/{}): {} ({})".format(ch+1,el.shape[1],pred, labels_mapping[pred]))
        LOG.debug("Label probabilities (%): {}\n".format(np.round(np.array(probs.detach().cpu().numpy())*100,2)))
    print("\tsingle-sample classification {}/{} done".format(num+1,len(data)))
    # # The plotting part is comented and must be checked and adapted in case it's needed to produce some figure
    # #plt.figure()
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax2.set_ylim(4, 20)
    # ax1.plot(range(1, len(el[-1])+1), el[-1], color="tab:blue", label="Ie/C")
    # ax1.set_ylabel("External current (a.u.)")
    # ax2.scatter(range(1, len(activity_spikes)+1), activity_spikes, color='tab:red', s=0.2, label="Activity")
    # ax2.set_ylim((-1,2))
    # ax2.set_yticks([0,1], ['rest', 'spike'])
    # ax2.set_ylabel("Neuronal response")
    # plt.title("Spikes from input as for panel {} \npred: {} ({}) with {}% probability".format(list(labels_mapping.keys())[num],pred,labels_mapping[pred],np.round(np.max(probs.cpu().numpy())*100,2)))
    # ax1.legend(loc=2)
    # ax2.legend(loc=1)
    # if save_fig:
    #     figure_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     plt.savefig(path_for_plots + "/input_{}_{}.pdf".format(list(labels_mapping.keys())[num],figure_datetime), dpi=300)
    #     plt.savefig(path_for_plots + "/input_{}_{}.png".format(list(labels_mapping.keys())[num],figure_datetime), dpi=300)
    # plt.show()
    # if save_fig:
    #     print("\tactivity plot {}/{} saved".format(num+1,len(list(labels_mapping.keys()))))
LOG.debug("---------------------------------------------------------------------------------------------------\n\n")
activity_classification["Letter"] = letter
activity_classification["Behaviour"] = behaviour
activity_classification["Probabilities"] = behaviour_probs
print("*** classification done ***")

print(activity_classification)

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