"""
This script allows to classify the MN neuron output 
spike patterns obtained from an NNI-optimized
network with trained MN neuron parameters.
The spike_classifier with NNI-optimized parameters
and pre-trained weights is used.

SPIKING ACTIVITY COLLECTED FROM BRAILLE CLASSIFICATION.

CHECK SETTINGS in settings.py

Fra, Vittorio,
Politecnico di Torino,
EDA Group,
Torino, Italy.
"""


import datetime
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sn
import torch
import torch.nn as nn

from data.loader import load_activity
from NNI.utils.utils import *
from settings import settings


### 1) various experiment settings #############################################

experiment_name = settings["experiment_name"]

experiment_id = settings["experiment_id"]
trained_layers_path = settings["trained_layers_path"]
best_test_id = settings["best_test_id"]

use_seed = settings["use_seed"]
multi_seed = settings["multi_seed"]

if use_seed:
    if multi_seed:
        seed_list = [42, 0, 1, 14, 2024, 16, 6, 999, 19]
    else:
        seed = 42
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
else:
    seed = None


experiment_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

################################################################################


### 2) data loading ############################################################

activity_dir = "./data/Activity"

activity_dataset = "GR_Braille"
activity_experiment = "GR_Braille_w"
experiment_number = 1

activity_path = os.path.join(activity_dir,activity_dataset,f"{activity_experiment}_{experiment_number}")

subset = "test"

lbl_string = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

activity_df = load_activity(activity_path, subset, lbl_string)

data = activity_df["Activity"].values
label = activity_df["Label"].values
stimulus_lbl = activity_df["Stimulus"].values

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

log_path = os.path.join("./logs/activity_classification/MN_activity",activity_dataset,f"{activity_experiment}_{experiment_number}")
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
LOG.debug("Activity classification from: {} ({})\n".format(activity_dataset+"/"+activity_experiment+"_"+str(experiment_number), subset))
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


### 4) Device resources usage ##################################################

### GPU
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
    LOG.debug("Single GPU manually selected. Setting up the simulation on {}".format("cuda:"+str(gpu_sel)))
    device = torch.device("cuda:"+str(gpu_sel))
    torch.cuda.set_per_process_memory_fraction(gpu_mem_frac, device=device)

### CPU
min_use = get_least_active_cores(num_cores=2)
LOG.debug("Selected CPU cores: {}".format(min_use))
limit_cpu_cores(min_use)

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



### various definitions ########################################################

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
        single_sample = torch.reshape(input_spikes, (1,input_spikes.shape[0],1)).to(device) # (batch_size, time, channels)
    else:
        rnd_idx = np.random.randint(0, input_spikes.shape[0])
        single_sample = torch.as_tensor(np.array(input_spikes[rnd_idx,:]), dtype=torch.float, device=device)
        single_sample = torch.reshape(single_sample, (1,single_sample.shape[0],1)).to(device) # (batch_size, time, channels)

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

LOG.debug("EXPERIMENT STARTED --- {}-{}-{} {}:{}:{}".format(
    experiment_datetime[:4],
    experiment_datetime[4:6],
    experiment_datetime[6:8],
    experiment_datetime[-6:-4],
    experiment_datetime[-4:-2],
    experiment_datetime[-2:])
    )

### Perform spiking patterns classification
print("*** classification started ***")
if multi_seed:

    for sd in range(settings["n_seed"]):

        seed = seed_list[sd]
        LOG.debug(f"\t --- \t seed ({sd+1}/{settings['n_seed']}): {seed} \t ---")

        activity_classification = pd.DataFrame()
        letter = []
        behaviour = []
        behaviour_probs = []
        n_spikes = []
        sparsity = []

        letter_repetitions = []
        behaviour_repetitions = []
        behaviour_probs_repetitions = []
        n_spikes_repetitions = []
        sparsity_repetitions = []

        non_zero_channels = 0
        non_zero_channels_repetitions = []
        zero_spikes_letters = []

        for num,el in enumerate(data):

            activity_spikes = el

            if el.nonzero().shape[0] > 0:
                n_spikes.append(int(el.sum().cpu().item()))
                sparsity.append(np.round(1-el.mean().cpu().item(),4))
                non_zero_channels += 1
                LOG.debug("Single-sample inference of 'active channel' {}/{}:".format(num+1,len(data)))
                LOG.debug("Stimulus label: {}".format(stimulus_lbl[num]))
                pred, probs = classify_spikes(activity_spikes, True, labels_mapping, trained_layers_path)
                letter.append(stimulus_lbl[num])
                behaviour.append(pred)
                behaviour_probs.append(np.round(np.array(probs.detach().cpu().numpy())*100,2))
                LOG.debug("Behaviour prediction: {} ({})".format(pred, labels_mapping[pred]))
                LOG.debug("Label probabilities (%): {}\n".format(np.round(probs.detach().cpu().numpy()*100,2)))
                print("\tsingle-sample classification of 'active channel' {}/{} done".format(num+1,len(data)))

                letter_repetitions.extend(letter)
                behaviour_repetitions.extend(behaviour)
                behaviour_probs_repetitions.extend(behaviour_probs)
                n_spikes_repetitions.extend(n_spikes)
                sparsity_repetitions.extend(sparsity)
            
            else:
                zero_spikes_letters.append(stimulus_lbl[num])
            
        non_zero_channels_repetitions.append(non_zero_channels)

        print(f"\t --- \t seed {sd+1}/{settings['n_seed']} ({seed}) done \t ---")
    
    LOG.debug(f"Number of channels with spiking activity for the different seeds (out of {len(data)}): {non_zero_channels_repetitions}")

    activity_classification["Letter"] = letter_repetitions
    activity_classification["Behaviour"] = behaviour_repetitions
    activity_classification["Probabilities"] = behaviour_probs_repetitions
    activity_classification["Spikes"] = n_spikes_repetitions
    activity_classification["Sparsity"] = behaviour_probs_repetitions
    df2pkl_path = os.path.join("./results/activity_classification/MN_activity",activity_dataset)
    create_directory(df2pkl_path)
    activity_classification.to_pickle(os.path.join(df2pkl_path,f"{activity_experiment}_{experiment_number}_{subset}_{settings['n_seed']}repetitions_{experiment_datetime}.pkl"))

else:
    
    activity_classification = pd.DataFrame()
    letter = []
    behaviour = []
    behaviour_probs = []
    n_spikes = []
    sparsity = []
    non_zero_channels = 0
    zero_spikes_letters = []
    
    for num,el in enumerate(data):

        activity_spikes = el

        if el.nonzero().shape[0] > 0:
            n_spikes.append(int(el.sum().cpu().item()))
            sparsity.append(np.round(1-el.mean().cpu().item(),4))
            non_zero_channels += 1
            LOG.debug("Single-sample inference of 'active channel' {}/{}:".format(num+1,len(data)))
            LOG.debug("Stimulus label: {}".format(stimulus_lbl[num]))
            pred, probs = classify_spikes(activity_spikes, True, labels_mapping, trained_layers_path)
            letter.append(stimulus_lbl[num])
            behaviour.append(pred)
            behaviour_probs.append(np.round(probs.detach().cpu().numpy()*100,2))
            LOG.debug("Behaviour prediction: {} ({})".format(pred, labels_mapping[pred]))
            LOG.debug("Label probabilities (%): {}\n".format(np.round(np.array(probs.detach().cpu().numpy())*100,2)))
            print("\tsingle-sample classification of 'active channel' {}/{} done".format(num+1,len(data)))
        
        else:
            zero_spikes_letters.append(stimulus_lbl[num])
    
    LOG.debug(f"Number of channels with spiking activity (out of {len(data)}): {non_zero_channels}")

    activity_classification["Letter"] = letter
    activity_classification["Behaviour"] = behaviour
    activity_classification["Probabilities"] = behaviour_probs
    activity_classification["Spikes"] = n_spikes
    activity_classification["Sparsity"] = sparsity
    df2pkl_path = os.path.join("./results/activity_classification/MN_activity",activity_dataset)
    create_directory(df2pkl_path)
    activity_classification.to_pickle(os.path.join(df2pkl_path,f"{activity_experiment}_{experiment_number}_{subset}_{experiment_datetime}.pkl"))

unique_lbl, unique_count = np.unique(zero_spikes_letters, return_counts=True)
count_zero_spikes, _ = np.unique(unique_count, return_counts=True)

if len(count_zero_spikes) == 1:
    if count_zero_spikes.item()*len(lbl_string) == data.shape[0]:
        LOG.debug("##### NO SPIKES #####")
        LOG.debug("---------------------------------------------------------------------------------------------------\n\n")
        print("*** No spikes ***")
else:
    ### Prepare the dataframe for the heatmap and plot it
    grouped = activity_classification[["Letter","Probabilities"]].groupby("Letter", as_index=False).mean()
    classified_activity_df = pd.DataFrame(index=range(len(lbl_string)), columns=range(len(list(labels_mapping.values()))))
    for ii in range(len(lbl_string)):
        for jj in range(len(list(labels_mapping.keys()))):
            classified_activity_df.iloc[ii,jj] = float(grouped[grouped["Letter"]==lbl_string[ii]]["Probabilities"].item()[-1][jj])
    classified_activity_df = classified_activity_df.apply(pd.to_numeric, errors='coerce')
    plt.figure(figsize=(16, 12))
    sn.heatmap(classified_activity_df.T,
               annot=True,
               fmt='.2f',
               cbar=False,
               square=False,
               cmap="YlOrBr"
               )
    plt.xticks(ticks=[ii+0.5 for ii in range(27)],labels=lbl_string, rotation=0)
    plt.yticks(ticks=[ii+0.5 for ii in range(20)],labels=labels_mapping.values(), rotation=0)
    plt.tight_layout()
    if settings["save_hm"]:
        if multi_seed:
            path_for_plots = os.path.join("./results/plots/activity_classification/MN_activity",activity_dataset,f"{activity_experiment}_{experiment_number}_{subset}_{settings['n_seed']}repetitions")
        else:
            path_for_plots = os.path.join("./results/plots/activity_classification/MN_activity",activity_dataset,f"{activity_experiment}_{experiment_number}_{subset}")
        create_directory(path_for_plots)
        path_to_save_fig = f'{path_for_plots}/hm_{experiment_id}_{best_test_id}_{experiment_datetime}'
        plt.savefig(path_to_save_fig+".png", dpi=300)
        plt.savefig(path_to_save_fig+".pdf", dpi=300)
        plt.close()
    else:
        plt.show()

    LOG.debug("---------------------------------------------------------------------------------------------------\n\n")
    print("*** classification done ***")

conclusion_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
print("EXPERIMENT DONE --- {}-{}-{} {}:{}:{}".format(
    conclusion_datetime[:4],
    conclusion_datetime[4:6],
    conclusion_datetime[6:8],
    conclusion_datetime[-6:-4],
    conclusion_datetime[-4:-2],
    conclusion_datetime[-2:])
    )

################################################################################