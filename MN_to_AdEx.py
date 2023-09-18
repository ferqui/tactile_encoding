import torch
from torch.utils.data import DataLoader
from torchnmf.nmf import NMF
import matplotlib.pyplot as plt
from time import localtime, strftime
from tqdm import trange
from utils import addToNetMetadata, addHeaderToMetadata, set_results_folder, generate_dict
import matplotlib

# matplotlib.pyplot.ioff()  # turn off interactive mode
import numpy as np
import os
from models import *
from datasets import load_data
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# from adexlif import AdexLIF
import time
import pickle
from torch.utils.tensorboard import SummaryWriter

Current_PATH = os.getcwd()
matplotlib.use('Agg')
# Set seed:
torch.manual_seed(0)

# ---------------------------- Input -----------------------------------
save_out = True  # Flag to save figures:
sweep_param_name = ['a', 'A1', 'A2', 'b', 'G', 'k1', 'k2']
sweep_ranges = [[-10, 10], [-100, 100], [-1000, 1000]]

MNclasses = {
    'A2B': {'a':0,'A1':0,'A2': 0,'b': 10,'G':50,'k1':200,'k2': 20,'gain': 1},
    'C2J': {'a':5,'A1':0,'A2': 0,'b': 10,'G':50,'k1':200,'k2': 20,'gain': 1},
    'K': {'a':30,'A1':0,'A2': 0,'b': 10,'G':50,'k1':200,'k2': 20,'gain': 1},
    'L': {'a':30,'A1':10,'A2': -0.6,'b': 10,'G':50,'k1':200,'k2': 20,'gain': 1},
    'M2O': {'a':5,'A1':10,'A2': -0.6,'b': 10,'G':50,'k1':200,'k2': 20,'gain': 1},
    'P2Q': {'a':5,'A1':5,'A2': -0.3,'b': 10,'G':50,'k1':200,'k2': 20,'gain': 1},
    'R': {'a':0,'A1':8,'A2': -0.1,'b': 10,'G':50,'k1':200,'k2': 20,'gain': 1},
    'S': {'a':5,'A1':-3,'A2': 0.5,'b': 10,'G':50,'k1':200,'k2': 20,'gain': 1},
    'T': {'a':-80,'A1':0,'A2': 0,'b': 10,'G':50,'k1':200,'k2': 20,'gain': 1},
}

run_with_fake_input = False
# ---------------------------- Parameters -----------------------------------
threshold = "enc"
run = "_3"

file_dir_params = 'parameters/'
param_filename = 'parameters_th' + str(threshold)
file_name_parameters = file_dir_params + param_filename + '.txt'
params = {}
with open(file_name_parameters) as file:
    for line in file:
        (key, value) = line.split()
        if key == 'time_bin_size' or key == 'nb_input_copies' or key == 'n_param_values' or key == 'min_range' or key == 'max_range':
            params[key] = int(value)
        else:
            params[key] = np.double(value)

# variable_range = np.linspace(params['min_range'], params['max_range'], params['n_param_values'])


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
            torch.nn.Linear(H, outputSize)
        )

    def forward(self, x):
        out = self.NLregression(x)
        return out


def sim_batch_training(dataset, device, neuron, model, training={}, list_loss=[],
                       results_dir=None, net=None, epoch=0,writer=None):
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None]*10)  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))

        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)

        model.reset()

        training['optimizer'].zero_grad()
        s_out_rec_adex = []
        for t in range(x_local.shape[1]):
            outputs = model(x_local[:, t, None, None]*1e3*10)
            # print('x_local_max',x_local[:, t, None, None].max())
            s_out_rec_adex.append(outputs)

        s_out_rec_train_adex = torch.stack(s_out_rec_adex, dim=1)
        s_out_rec_train_adex = torch.permute(s_out_rec_train_adex, (3, 0, 2, 4, 1))
        s_out_rec_train_adex = torch.flatten(s_out_rec_train_adex, start_dim=0, end_dim=1)
        # print('total spikes (adex,MN)',s_out_rec_train_adex.sum(),s_out_rec_train.sum())
        reg = (s_out_rec_train.sum()-s_out_rec_train_adex.sum())**2
        loss = training['criterion'](s_out_rec_train_adex,s_out_rec_train)*10000 + reg
        # get gradients w.r.t to parameters
        loss.backward()
        list_loss.append([loss.item()])
        # update parameters
        training['optimizer'].step()
        parameters_adex = {'Vr' : model.Vr.item(),
                           'Vth' : model.Vth.item(),
                           'Vrh' : model.Vrh.item(),
                            'Vreset' : model.Vreset.item(),
                            'delta_T' : model.delta_T.item(),
                            'a' : model.a.item(),
                            'b' : model.b.item(),
                            'R' : model.R.item(),
                            'taum' : model.taum.item(),
                            'tauw' : model.tauw.item()}
        # nn.Parameter(torch.tensor(-30.),requires_grad=True) # Firing threshold
        # self.Vrh = nn.Parameter(torch.tensor(-50.),requires_grad=True)
        # self.Vreset = nn.Parameter(torch.tensor(-51.),requires_grad=True) # reset potential
        # self.delta_T = nn.Parameter(torch.tensor(2.),requires_grad=True) # Sharpness of the exponential term
        # self.a = nn.Parameter(torch.tensor(0.5),requires_grad=True) # Adaptation-Voltage coupling
        # self.b = nn.Parameter(torch.tensor(7.0),requires_grad=True) # Spike-triggered adaptation current
        # self.R = nn.Parameter(torch.tensor(0.5),requires_grad=True) # Resistance
        # self.taum = nn.Parameter(torch.tensor(5.),requires_grad=True) # membrane time scale
        # self.tauw =}
    return list_loss, net,parameters_adex,reg


def sim_batch_testing(dataset, device, neuron, training, model, list_loss=[],
                      final=False, results_dir=None, net=None, neuron_id=4, epoch=0,writer = None):
    with torch.no_grad():
        for x_local, y_local in dataset:
            x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
            neuron.reset()
            s_out_rec = []
            for t in range(x_local.shape[1]):
                out = neuron(x_local[:, t, None, None]*10)  # shape: n_batches x 1 fanout x n_param_values x n_channels
                s_out_rec.append(out)
            s_out_rec_test = torch.stack(s_out_rec, dim=1)
            s_out_rec_test = torch.permute(s_out_rec_test, (3, 0, 2, 4, 1))
            s_out_rec_test = torch.flatten(s_out_rec_test, start_dim=0, end_dim=1)

            ### s_out_rec_train shape:  trial x variable x fanout x channels x time
            s_out_rec_adex = []
            model.reset()
            for t in range(x_local.shape[1]):
                outputs = model(x_local[:, t, None, None]*1e3*10)
                s_out_rec_adex.append(outputs)
            s_out_rec_test_adex = torch.stack(s_out_rec_adex, dim=1)
            s_out_rec_test_adex = torch.permute(s_out_rec_test_adex, (3, 0, 2, 4, 1))
            s_out_rec_test_adex = torch.flatten(s_out_rec_test_adex, start_dim=0, end_dim=1)
            # reg = (s_out_rec_test.sum() - s_out_rec_test_adex.sum()) ** 2
            loss = training['criterion'](s_out_rec_test_adex, s_out_rec_test)
            list_loss.append([loss.item()])
        spikes_adex = torch.where(s_out_rec_test_adex[0, 0, :, :] > 0)
        spikes = torch.where(s_out_rec_test[0, 0, :, :] > 0)
        aaa = plt.scatter(spikes_adex[1].clone().detach().cpu(), spikes_adex[0].clone().detach().cpu()+0.5, color='r',label='AdEx',marker='x')
        bbb = plt.scatter(spikes[1].clone().detach().cpu(), spikes[0].clone().detach().cpu(), color='b',label='MN',marker='+')
        plt.legend(handles=[aaa,bbb])
        plt.xlim([0, x_local.shape[1]])
        # print('xlocal_sum',x_local.sum())
        # print('adex',s_out_rec_test_adex.sum(),'MN', s_out_rec_test.sum())

        spikes_adex = torch.where(s_out_rec_test_adex[0, 0, 20, :] > 0)
        spikes = torch.where(s_out_rec_test[0, 0, 20,:] > 0)
        if writer is not None:
            writer.add_figure('test', plt.gcf(), epoch)
        if len(spikes_adex[0] > 1) and len(spikes[0] > 1):
            diff_adex = torch.diff(spikes_adex[0].clone().detach().cpu())
            diff_spikes = torch.diff(spikes[0].clone().detach().cpu())
            fig1, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax2.plot(x_local[0, :, 20].clone().detach().cpu(), color='k', label='Input')
            ax1.plot(spikes[0][1:].clone().detach().cpu(), diff_spikes, '+',color='b', label='MN')
            ax1.plot(spikes_adex[0][1:].clone().detach().cpu(), diff_adex, 'x',color='r', label='AdEx')
            plt.xlim([0, x_local.shape[1]])

            fig1.legend()
            if writer is not None:
                writer.add_figure('test_diff', plt.gcf(), epoch)
    return list_loss,net




def MI_neuron_params(neuron_param_values, name_param_sweeped, extremes_sweep, MNclass):
    # Set results folder:

    iscuda = torch.cuda.is_available()
    # # Forcing CPU
    # iscuda = False
    # print('NOTE: Forced CPU')

    if run_with_fake_input:
        results_dir = 'results/controlled_stim/'
        if not (os.path.isdir(results_dir)):
            os.mkdir(results_dir)
    else:
        results_dir = set_results_folder([exp_id,MNclass, name_param_sweeped, str(extremes_sweep)])
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
    upsample_fac = 1
    file_name = "data/data_braille_letters_all.pkl"
    data, labels, _, _, _, _ = load_data(file_name, upsample_fac)
    nb_channels = data.shape[-1]

    x_train, x_test, y_train, y_test = train_test_split(
        data.cpu(), labels.cpu(), test_size=0.2, shuffle=True, stratify=labels.cpu()
    )

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)
    params['nb_channels'] = nb_channels
    params['labels'] = labels
    # params['data_steps'] = data_steps

    # Network parameters
    nb_input_copies = params['nb_input_copies']
    nb_inputs = params['nb_channels'] * nb_input_copies
    nb_hidden = 450
    nb_outputs = len(torch.unique(params['labels']))

    # Learning parameters
    # nb_steps = params['data_steps']
    nb_epochs = int(params['nb_epochs'])

    # Neuron parameters
    tau_mem = params['tau_mem']  # ms
    tau_syn = tau_mem / params['tau_ratio']

    encoder_weight_scale = params['scale']
    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = params['weight_scale_factor'] * fwd_weight_scale

    ###########################################
    ##                Network                ##
    ###########################################

    # a = torch.empty((nb_inputs,))
    n_param_values = params['n_param_values']
    # a = torch.Tensor(np.linspace(-10, 10, n_param_values)).to(device)
    chosen_class = MNclasses[MNclass]
    tensor_params = {}
    for key in chosen_class.keys():
        tensor_params[key] = torch.Tensor([chosen_class[key]]).to(device)

    # varying_element = tensor_params[name_param_sweeped]

    # a = torch.Tensor(neuron_param_values['a']).to(device)
    # # nn.init.normal_(
    # #     a, mean=MNparams_dict['A2B'][0], std=fwd_weight_scale / np.sqrt(nb_inputs))
    #
    # A1 = torch.Tensor(neuron_param_values['A1']).to(device)
    #
    # A2 = torch.Tensor(neuron_param_values['A2']).to(device)
    dt=1/100
    fanout = 1  # number of output neurons from the linear expansion
    # TODO: Change input parameters list
    neuron = MN_neuron_IT(params['nb_channels'], fanout, 1,
                                 tensor_params['a'],
                                 tensor_params['A1'],
                                 tensor_params['A2'],
                                 tensor_params['b'],
                                 tensor_params['G'],
                                 tensor_params['k1'],
                                 tensor_params['k2'],
                                 train=False,
                                 dt=dt)
    print('Neuron parameters(a,A1,A2,b,G,k1,k2): ', neuron.a[0,0,0,0].item(), neuron.A1[0,0,0,0].item(), neuron.A2[0,0,0,0].item(), neuron.b[0,0,0,0].item(), neuron.G[0,0,0,0].item(), neuron.k1[0,0,0,0].item(), neuron.k2[0,0,0,0].item())
    batch_size = int(params['batch_size'])

    inputDim = 1  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    learningRate = params['learningRate']  # 10#0.01

    neuron_id = int(params['neuron_id'])

    rank_NMF = int(params['rank_NMF'])
    range_weight_init = 10
    model = AdexLIF(n_in=1,n_out=fanout,dt=dt,params_n=1,channels=params['nb_channels'])
    # model.linear.weight = torch.nn.Parameter(model.linear.weight*range_weight_init)

    coeff = [p for p in model.parameters()][0][0]
    coeff = coeff.clone().detach()

    # TODO: Check this
    if iscuda:
        pin_memory = True
        num_workers = 0
    else:
        pin_memory = False
        num_workers = 0

    # The log softmax function across output units
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                          generator=torch.Generator(device=device))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                         generator=torch.Generator(device=device))
    # pbar = trange(nb_epochs)
    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    optimizer = torch.optim.Adamax(model.parameters(), lr=learningRate)
    params['optimizer'] = optimizer.__class__.__name__
    # print('Training classifier with {} optimizer'.format(params['optimizer']))

    # Store parameters to metadata file:
    for key in params.keys():
        addToNetMetadata(metadatafilename, key, params[key])

    header = 'Neuron params'
    for key in neuron_param_values.keys():
        addToNetMetadata(metadatafilename, key, neuron_param_values[key], header=header)
        header = ''

    t_start = time.time()
    net = None
    pbar = trange(nb_epochs)
    list_loss_train = []
    list_loss_test = []
    writer = SummaryWriter(comment="MN_to_AdEx_class"+str(MNclass))
    for e in pbar:
        local_loss = []
        H_train = []
        H_test = []
        accs = []  # accs: mean training accuracies for each batch
        # print('Epoch', e)
        list_loss_train, net,adexparam,reg = sim_batch_training(dataset=dl_train, device=device, neuron=neuron,model= model,
                                            training={'optimizer': optimizer, 'criterion': criterion}, list_loss=list_loss_train,
                                            results_dir=results_dir, net=net, epoch=e)
        list_loss_test, net = sim_batch_testing(dataset=dl_test, device=device, neuron=neuron,training={'optimizer': optimizer, 'criterion': criterion}, model=model, list_loss=list_loss_test,
                                         results_dir=results_dir, net=net, neuron_id=neuron_id, epoch=e,writer=writer)
        pbar.set_postfix_str("Loss train: " + str(np.round(list_loss_train[-1][0], 10)) + '. Loss test: ' + str(
            np.round(list_loss_test[-1][0], 10)) + ', reg_train: ' + str(reg.item()))
        writer.add_scalar("Loss/train", list_loss_train[-1][0], global_step=e)
        writer.add_scalar("Loss/test", list_loss_test[-1][0], global_step=e)
        for key in adexparam.keys():
            writer.add_scalar(key, adexparam[key], global_step=e)
    train_duration = time.time() - t_start
    addToNetMetadata(metadatafilename, 'sim duration (sec)', train_duration)




if __name__ == "__main__":
    print('Current path',Current_PATH)
    for MNclass in MNclasses:
        print('-------------------------------------------')
        print('Class {}'.format(MNclass))
        # Generate dictionary with parameter values:
        variable_range = torch.linspace(0, 0,
                                        1)
        name_param = 'G'
        dict_keys = generate_dict(name_param, variable_range,force_param_dict=MNclasses[MNclass])

        # Run mutual information analysis
        MI_neuron_params(dict_keys, name_param, [0,0], MNclass)
