import torch
from torch.utils.data import DataLoader
from torchnmf.nmf import NMF
import matplotlib.pyplot as plt
from time import localtime, strftime
from tqdm import trange
from utils import addToNetMetadata, addHeaderToMetadata, set_results_folder, generate_dict
import matplotlib

matplotlib.pyplot.ioff()  # turn off interactive mode
import numpy as np
import os
from models import *
from datasets import load_analog_data
# from adexlif import AdexLIF
import time
import pickle
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
                       results_dir=None, net=None, epoch=0):
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None] * 1000)  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))

        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)

        model.reset()

        training['optimizer'].zero_grad()
        s_out_rec_adex = []
        for t in range(x_local.shape[1]):
            outputs = model(x_local[:, t, None, None] * 1000)
            s_out_rec_adex.append(outputs)
        s_out_rec_train_adex = torch.stack(s_out_rec_adex, dim=1)
        s_out_rec_train_adex = torch.permute(s_out_rec_train_adex, (3, 0, 2, 4, 1))
        s_out_rec_train_adex = torch.flatten(s_out_rec_train_adex, start_dim=0, end_dim=1)
        loss = training['criterion'](s_out_rec_train_adex,s_out_rec_train)
        # get gradients w.r.t to parameters
        loss.backward()
        list_loss.append([loss.item()])
        # update parameters
        training['optimizer'].step()
        parameters_adex = {'Vr' : model.Vr.item(),
                           'Vth' : model.Vth.item(),
                           'Vrh' : model.Vrh.item()}
        # nn.Parameter(torch.tensor(-30.),requires_grad=True) # Firing threshold
        # self.Vrh = nn.Parameter(torch.tensor(-50.),requires_grad=True)
        # self.Vreset = nn.Parameter(torch.tensor(-51.),requires_grad=True) # reset potential
        # self.delta_T = nn.Parameter(torch.tensor(2.),requires_grad=True) # Sharpness of the exponential term
        # self.a = nn.Parameter(torch.tensor(0.5),requires_grad=True) # Adaptation-Voltage coupling
        # self.b = nn.Parameter(torch.tensor(7.0),requires_grad=True) # Spike-triggered adaptation current
        # self.R = nn.Parameter(torch.tensor(0.5),requires_grad=True) # Resistance
        # self.taum = nn.Parameter(torch.tensor(5.),requires_grad=True) # membrane time scale
        # self.tauw =}
    return list_loss, net,parameters_adex


def sim_batch_testing(dataset, device, neuron, training, model, list_loss=[],
                      final=False, results_dir=None, net=None, neuron_id=4, epoch=0):

    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None] * 1000)  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
        s_out_rec_test = torch.stack(s_out_rec, dim=1)
        s_out_rec_test = torch.permute(s_out_rec_test, (3, 0, 2, 4, 1))
        s_out_rec_test = torch.flatten(s_out_rec_test, start_dim=0, end_dim=1)

        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        s_out_rec_adex = []
        model.reset()
        for t in range(x_local.shape[1]):
            outputs = model(x_local[:, t, None, None] * 1000)
            s_out_rec_adex.append(outputs)
        s_out_rec_test_adex = torch.stack(s_out_rec_adex, dim=1)
        s_out_rec_test_adex = torch.permute(s_out_rec_test_adex, (3, 0, 2, 4, 1))
        s_out_rec_test_adex = torch.flatten(s_out_rec_test_adex, start_dim=0, end_dim=1)
        loss = training['criterion'](s_out_rec_test_adex, s_out_rec_test)
        list_loss.append([loss.item()])
    return list_loss,net


def sim_batch_final(dataset, device, neuron, varying_element, rank_NMF, model, list_mi=[], results_dir=None, net=None,
                    neuron_id=4, epoch=0):
    list_mi_local = []
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None] * 1000)  # shape: n_batches x 1 fanout x n_param_values x n_channels
            s_out_rec.append(out)
        s_out_rec_train = torch.stack(s_out_rec, dim=1)
        s_out_rec_train = torch.permute(s_out_rec_train, (3, 0, 2, 4, 1))
        ### s_out_rec_train shape:  trial x variable x fanout x channels x time
        label = torch.ones([s_out_rec_train.shape[1], varying_element.shape[0]]) * varying_element
        s_out_rec_train = torch.flatten(s_out_rec_train, start_dim=0, end_dim=1)
        label = label.T.flatten()[:, None]
        label_min = label.min()
        label_diff = label.max() - label.min()
        label_norm = (label - label_min) / (label_diff)
        H = torch.zeros([s_out_rec_train.shape[2], s_out_rec_train.shape[0], rank_NMF])
        V_matrix = s_out_rec_train[:, 0, neuron_id, :]
        # if net is None:
        net = NMF(V_matrix.shape, rank=rank_NMF)
        net.fit(V_matrix)
        H[neuron_id] = net.H
        predicted = model(H[neuron_id])
        output_vs_label = [predicted.clone().detach(),
                           label.clone().detach()]
        outputs_norm = (predicted - label_min) / (label_diff)
        acc = ((outputs_norm - label_norm) ** 2).sum() / x_local.shape[0]
        label_unique = torch.unique(label)
        predicted_int = predicted.type(torch.int)
        predicted_range = torch.unique(predicted_int)

        pdf_x1x2 = torch.zeros([len(label_unique), len(predicted_range)])
        for trial_idx in range(len(predicted_int)):
            lab_pos = torch.where(label_unique == label[trial_idx])[0]
            pred_pos = torch.where(predicted_range == predicted_int[trial_idx])[0]
            pdf_x1x2[lab_pos, pred_pos] += 1

        num_occ = torch.sum(pdf_x1x2)
        pdf_x1 = torch.sum(pdf_x1x2, dim=1) / num_occ  # to check
        pdf_x2 = torch.sum(pdf_x1x2, dim=0) / num_occ
        pdf_x1x2 = pdf_x1x2 / num_occ
        f = plt.figure()
        plt.imshow(pdf_x1x2.clone().detach().cpu())
        plt.xticks([i for i in range(len(predicted_range))], np.array(predicted_range.cpu()))
        plt.yticks([i for i in range(len(label_unique))], np.array(label_unique.cpu()))
        plt.xlabel('Predicted')
        plt.ylabel('Label')
        plt.title('Prob matrix')
        plt.colorbar()
        if save_out:
            f.savefig(results_dir + 'pdf_joint.pdf', format='pdf')

        f = plt.figure()
        plt.imshow(x_local[:, neuron_id, :].clone().detach().cpu(), aspect='auto')
        plt.title('XLOCAL')
        if save_out:
            f.savefig(results_dir + 'xlocal.pdf', format='pdf')

        f = plt.figure()
        plt.imshow(s_out_rec_train[:, 0, neuron_id, :].clone().detach().cpu(), aspect='auto')
        # plt.ylim([0, 128*2])
        plt.title('INPUT NMF')
        plt.xlabel('Time')
        plt.ylabel('TrialxVariable')
        if save_out:
            f.savefig(results_dir + 'Input_nmf.pdf', format='pdf')

        # s_out_rec_train shape:  (trial x variable) x fanout x channels x time
        # print('s_out_train shape', s_out_rec_train.shape)
        f = plt.figure()
        plt.imshow(H[neuron_id].clone().detach().cpu(), aspect='auto', interpolation='nearest')
        # plt.ylim([0, 128*2])
        plt.title('H NMF')
        plt.xlabel('Time')
        plt.ylabel('TrialxVariable')
        if save_out:
            f.savefig(results_dir + 'H.pdf', format='pdf')

        f = plt.figure()
        plt.title('W NMF')
        plt.imshow(net.W.clone().detach().cpu(), aspect='auto', interpolation='nearest')
        plt.xlabel('Time')
        plt.ylabel('Rank')
        if save_out:
            f.savefig(results_dir + 'W_nmf.pdf', format='pdf')

        if run_with_fake_input:
            # Store V input:
            with open(results_dir + 'W.pickle', 'wb') as f:
                pickle.dump(net.W.clone().detach().cpu()(), f)

        f = plt.figure()
        plt.imshow(net().clone().detach().cpu(), aspect='auto', interpolation='nearest')
        # plt.ylim([0, 128*2])
        plt.title('OUT NMF')
        plt.xlabel('Time')
        plt.ylabel('TrialxVariable')
        if save_out:
            f.savefig(results_dir + 'Out_nmf.pdf', format='pdf')

        f = plt.figure()
        plt.imshow(torch.concat(output_vs_label, dim=1).clone().detach().cpu(), aspect='auto', cmap='seismic',
                   interpolation='nearest')
        plt.colorbar()
        plt.title('OUTPUT vs LABEL')
        # plt.xlabel('Output|Label')
        plt.ylabel('TrialxVariable')
        plt.xticks([0, 1], ['Output', 'Label'])
        if save_out:
            f.savefig(results_dir + 'Predicted_vs_label.pdf', format='pdf')

        if run_with_fake_input:
            # Store Out nmf input:
            with open(results_dir + 'Out_nmf.pickle', 'wb') as f:
                pickle.dump(net().clone().detach().cpu(), f)
            # Store Out classifier:
            with open(results_dir + 'Out_classifier.pickle', 'wb') as f:
                pickle.dump(predicted.clone().detach().cpu(), f)
            # Store Learned W coefficients:
            with open(results_dir + 'w_classifier.pickle', 'wb') as f:
                coeff = [p for p in model.parameters()][0][0]
                coeff = coeff.clone().detach()
                pickle.dump(coeff[:, None].clone().detach().cpu(), f)
            # Store joint probability matrix:
            with open(results_dir + 'pdf_x1x2.pickle', 'wb') as f:
                pickle.dump(pdf_x1x2.clone().detach().cpu(), f)
        mi = torch.zeros(1)
        for el1_idx, pdf_x1_el in enumerate(pdf_x1):
            for el2_idx, pdf_x2_el in enumerate(pdf_x2):
                mi += pdf_x1x2[el1_idx, el2_idx] * torch.log2(
                    (pdf_x1x2[el1_idx, el2_idx] / (pdf_x1_el * pdf_x2_el)) + 1E-10)
                # print('mutual information', mi)
                # plt.figure()
                # plt.imshow(pdf_x1x2,aspect='auto')
                # plt.title('PDF')
                # plt.figure()
                # plt.plot(label, predicted_int)
                # plt.show()
                list_mi.append([mi.item(), epoch])
    # list_mi.append(torch.mean(torch.Tensor(list_mi_local)))
    return list_mi, net


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
    upsample_fac = 5
    file_name = "data/data_braille_letters_digits.pkl"
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
    # a = torch.Tensor(np.linspace(-10, 10, n_param_values)).to(device)
    chosen_class = MNclasses['A2B']
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

    list_loss = []
    list_mi = []
    t_start = time.time()
    H_initial = None
    # net = NMF((), rank=rank_NMF, H=H_initial)
    net = None
    pbar = trange(nb_epochs)
    print('eeeeee')
    list_loss_train = []
    list_loss_test = []
    for e in pbar:
        local_loss = []
        H_train = []
        H_test = []
        accs = []  # accs: mean training accuracies for each batch
        # print('Epoch', e)
        list_loss_train, net,adexparam = sim_batch_training(dataset=dl_train, device=device, neuron=neuron,model= model,
                                            training={'optimizer': optimizer, 'criterion': criterion}, list_loss=list_loss_train,
                                            results_dir=results_dir, net=net, epoch=e)
        list_loss_test, net = sim_batch_testing(dataset=dl_test, device=device, neuron=neuron,training={'optimizer': optimizer, 'criterion': criterion}, model=model, list_loss=list_loss_test,
                                         results_dir=results_dir, net=net, neuron_id=neuron_id, epoch=e)
        pbar.set_postfix_str("Loss train: " + str(np.round(list_loss_train[-1], 10)) + '. Loss test: ' + str(
            np.round(list_loss_test[-1], 10)) + str(adexparam))
    train_duration = time.time() - t_start
    addToNetMetadata(metadatafilename, 'sim duration (sec)', train_duration)

    list_mi, net = sim_batch_final(dl_test, device, neuron, varying_element, rank_NMF, model, list_mi=list_mi,
                                   results_dir=results_dir, neuron_id=neuron_id, epoch=e)
    plt.close('all')
    # Store data:
    # Loss:
    list_loss = np.array(list_loss)
    list_mi = np.array(list_mi)
    with open(results_dir + 'Loss.pickle', 'wb') as f:
        pickle.dump(list_loss, f)
    # Mi:
    with open(results_dir + 'MI.pickle', 'wb') as f:
        pickle.dump(list_mi, f)
    #
    # fig = plt.figure()
    # plt.plot(list_loss)
    # plt.xlabel('Epochs x Trials')
    # plt.ylabel('Loss')
    # if save_out:
    #     fig.savefig(results_dir+'Loss.pdf', format='pdf')
    #
    # fig = plt.figure()
    # plt.plot(list_mi[:,0])
    # plt.xlabel('Epochs x Trials')
    # plt.ylabel('MI')
    # if save_out:
    #     fig.savefig(results_dir+'MI.pdf', format='pdf')


if __name__ == "__main__":
    print('Current path',Current_PATH)
    for MNclass in MNclasses:
        for name_param in sweep_param_name:
            for variable_range_extremes in sweep_ranges:
                print('-------------------------------------------')
                print('Class {}: Sweeping {} from {} to {}'.format(MNclass, name_param, variable_range_extremes[0],
                                                                    variable_range_extremes[1]))
                # Generate dictionary with parameter values:
                variable_range = torch.linspace(variable_range_extremes[0], variable_range_extremes[1],
                                                params['n_param_values'])
                dict_keys = generate_dict(name_param, variable_range,force_param_dict=MNclasses[MNclass])

                # Run mutual information analysis
                MI_neuron_params(dict_keys, name_param, variable_range_extremes, MNclass)
