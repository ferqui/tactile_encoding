"""
Once the weights for the 'optimal' (in whatever sense) network are available,
test results (with statistics and a confusion matrix) can be produced with
this code by properly setting the 'experiment_id' (i.e. NNI experiment) and
the 'best_test_id' (i.e. the trial with best test form the NNI experiment)
variable.

GPU, seed and saving settings are available and can be customized.

Fra, Vittorio,
Politecnico di Torino,
EDA Group,
Torino, Italy.
"""

import numpy as np
import pandas as pd

import os
import datetime

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#from tactile_encoding.utils.utils import check_cuda, value2index, create_directory, load_layers
from utils.utils import check_cuda, create_directory


experiment_id = "83iopuwk"
best_test_id = "s0yEm"

# set up CUDA device
global device
device = check_cuda(gpu_sel=1, gpu_mem_frac=0.3)

global use_seed
use_seed = False

global save_fig
save_fig = True

global seed
if use_seed:
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    seed = None

global labels_mapping
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


def run_snn(params, inputs, layers):

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


def load_layers(file, map_location, requires_grad=True, variable=False):

    if variable:
        lays = file
        for ii in lays:
            ii.requires_grad = requires_grad
    else:
        lays = torch.load(file, map_location=map_location)
        for ii in lays:
            ii.requires_grad = requires_grad
    return lays


def build_and_test(features,
                   attributes,
                   layers_file,
                   device=device,
                   N=10,
                   report=True):
    
    with open(os.path.expanduser("~/nni-experiments/spike_classifier/{}/trials/{}/parameter.cfg".format(layers_file,best_test_id)), "r") as handle:
        read_dict = handle.read()
    params = eval(read_dict) # the dictionary from NNI
    params = params["parameters"]
    params["batch_size"] = 64

    # prepare data selection
    global name
    name = ""
    data_features = features #[original, fixed_length, noise, jitter]
    data_attributes = attributes #["original", "fix_len", "noisy", "temp_jitter"]
    for num,el in enumerate(list(np.where(np.array(data_features)==True)[0])):
        name += "{} ".format(data_attributes[el])
    name = name[:-1]
    name = name.replace(" ","_")

    ds_test = torch.load("./dataset_splits/{}/{}_ds_test.pt".format(name,name), map_location=device)

    data_steps = len(next(iter(ds_test))[0])

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

    if report:
        path = './results/reports/test_spike_classifier/{}'.format(name)
        create_directory(path)
        report_path = path + "/{}".format(layers_file)

        trial_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Spiking network
    layers_path = "./results/layers/fix_len_noisy_temp_jitter/{}.pt".format(layers_file)
    layers = load_layers(layers_path, map_location=device)

    test_N = []

    for ii in range(N):
        
        test_acc, _, _ = compute_classification_accuracy(params, ds_test, layers=layers, label_probabilities=True, shuffle=True)

        if report:
            with open(report_path, 'a') as f:
                f.write("Test {}/{} performed on: {}-{}-{} {}:{}:{}\n".format(ii+1,N,
                    trial_datetime[:4],
                    trial_datetime[4:6],
                    trial_datetime[6:8],
                    trial_datetime[-6:-4],
                    trial_datetime[-4:-2],
                    trial_datetime[-2:]))
                f.write("\tAccuracy: {} (%)".format(np.round(test_acc*100,2)))
                f.write('\n')
        
        test_N.append(test_acc)
        print("Test {}/{}: {}%".format(ii+1,N,np.round(test_acc*100,2)))

    if report:
        with open(report_path, 'a') as f:
            f.write('\n')
            f.write("Min. test accuracy: {}%\n".format(np.round(np.min(test_N)*100,2)))
            f.write("Max. test accuracy: {}%\n".format(np.round(np.max(test_N)*100,2)))
            f.write("Mean test accuracy: {}%\n".format(np.round(np.mean(test_N)*100,2)))
            f.write("Median test accuracy: {}%\n".format(np.round(np.median(test_N)*100,2)))
            f.write("Std. Dev. test accuracy: {}%\n".format(np.round(np.std(test_N)*100,2)))
            f.write("---------------------------------------------------------------------------------------------------\n")
            f.write('\n')
    
    ConfusionMatrix(params, ds_test, save_fig, layers, list(labels_mapping.keys()))


def compute_classification_accuracy(params, dataset, layers=None, label_probabilities=False, shuffle=False):
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
            spks_out, _, _ = run_snn(params=params, inputs=x_local, layers=layers)
        else:
            spks_out, _, _ = run_snn(params=params, inputs=x_local, layers=layers)
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


def ConfusionMatrix(params, dataset, save, layers=None, labels=None, use_seed=use_seed):
        
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
            spks_out, _, _ = run_snn(params=params, inputs=x_local, layers=layers)
        else:
            spks_out, _, _ = run_snn(params=params, inputs=x_local, layers=layers)
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
    plt.title("Accuracy from confusion matrix: {:.2f}% +- {:.2f}%\n".format(np.median(accs) * 100, np.std(accs)*100))
    plt.xlabel('\nPredicted')
    plt.ylabel('True\n')
    plt.xticks(rotation=0)
    if save:
        path_for_plots = f'./results/plots/{name}'
        isExist_record = os.path.exists(path_for_plots)
        if not isExist_record:
            os.makedirs(path_for_plots)
        #path_to_save_fig = f'{path_for_plots}/generation_{generation+1}_individual_{best_individual+1}'
        path_to_save_fig = f'{path_for_plots}/{experiment_id}_{best_test_id}_cm'
        if use_trainable_tc:
            path_to_save_fig = f'{path_to_save_fig}_train_tc'
        if use_trainable_out:
            path_to_save_fig = f'{path_to_save_fig}_train_out'
        if use_dropout:
            path_to_save_fig = f'{path_to_save_fig}_dropout'
        #path_to_save_fig = f'{path_to_save_fig}_cm.png'
        path_to_save_fig = f'{path_to_save_fig}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
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


####################################################################################################################################


# Specify what kind of data to use
original = False
fixed_length = not original
noise = True
jitter = True

features = [original, fixed_length, noise, jitter]
attributes = ["original", "fix_len", "noisy", "temp_jitter"]

build_and_test(features, attributes, layers_file=experiment_id)
