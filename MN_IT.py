import torch
from torch.utils.data import TensorDataset,DataLoader
from torchnmf.nmf import NMF
import matplotlib.pyplot as plt
from time import localtime, strftime
from tqdm import trange
from utils import addToNetMetadata, addHeaderToMetadata, set_results_folder, generate_dict
import matplotlib
from torch.utils.tensorboard import SummaryWriter

matplotlib.pyplot.ioff()  # turn off interactive mode
import numpy as np
import os
import models
from datasets import load_analog_data, load_data
from sklearn.model_selection import train_test_split
from experiments.plot_MN_IT import plot_the_data
import time
import pickle
Current_PATH = os.getcwd()
matplotlib.use('Agg')
# Set seed:
torch.manual_seed(0)

# ---------------------------- Input -----------------------------------
save_out = True  # Flag to save figures:
sweep_param_name = ['gain','a', 'A1', 'A2', 'b', 'G', 'k1', 'k2']
# sweep_param_name = ['b', 'G', 'k1', 'k2']
letters = ['A','B']
sweep_ranges = [[-10, 10]]#, [-100, 100], [-1000, 1000]]

MNclasses = {
    #'A2B': {'a':0,'A1':0,'A2': 0},
    'C2J': {'a':5,'A1':0,'A2': 0},
    # 'K': {'a':30,'A1':0,'A2': 0},
    # 'L': {'a':30,'A1':10,'A2': -0.6},
    # 'M2O': {'a':5,'A1':10,'A2': -0.6},
    # 'P2Q': {'a':5,'A1':5,'A2': -0.3},
    # 'R': {'a':0,'A1':8,'A2': -0.1},
    # 'S': {'a':5,'A1':-3,'A2': 0.5},
    # 'T': {'a':-80,'A1':0,'A2': 0}
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



def sim_batch_training(dataset, device, neuron, varying_element, rank_NMF, model, training={}, list_loss=[],
                       results_dir=None, net=None, neuron_id=4, epoch=0):
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None])  # shape: n_batches x 1 fanout x n_param_values x n_channels
            # print(out.shape)
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
        # V_matrix = s_out_rec_train[:, 0, neuron_id, :]
        # print('s_out_rec_train.shape',s_out_rec_train.shape)
        #s_out_rec_train = torch.permute(s_out_rec_train, (0, 1, 3, 2))
        V_matrix_flattn = torch.flatten(s_out_rec_train,start_dim=2,end_dim=3)
        V_matrix = V_matrix_flattn[:, 0, :]
        # print('V_matrix_flattn.shape',V_matrix_flattn.shape)

        net = NMF(V_matrix.shape, rank=rank_NMF)
        net.fit(V_matrix)
        H[neuron_id] = net.H
        training['optimizer'].zero_grad()
        outputs = model(H[neuron_id])
        outputs_norm = (outputs - label_min) / (label_diff)
        loss = training['criterion'](outputs, label)
        # get gradients w.r.t to parameters
        loss.backward()
        list_loss.append([loss.item(), epoch])
        # update parameters
        training['optimizer'].step()
    return list_loss, net


def sim_batch_testing(dataset, device, neuron, varying_element, rank_NMF, model, list_mi=[], list_spikecount = {'mean':[], 'std':[]},
                      final=False, results_dir=None, net=None, neuron_id=4, epoch=0):
    list_mi_local = []
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None])  # shape: n_batches x 1 fanout x n_param_values x n_channels
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
        # selected_neuron_id = 4
        # net = None
        # s_out_rec_train = torch.permute(s_out_rec_train, (0, 1, 3, 2))

        V_matrix_flattn = torch.flatten(s_out_rec_train, start_dim=2, end_dim=3)
        V_matrix = V_matrix_flattn[:, 0, :]

        sum_over_time = torch.sum(V_matrix, dim=1)
        list_spikecount['mean'].append(torch.mean(sum_over_time).item())
        list_spikecount['std'].append(torch.std(sum_over_time).item())

        net = NMF(V_matrix.shape, rank=rank_NMF)
        net.fit(V_matrix)
        H[neuron_id] = net.H
        predicted = model(H[neuron_id])
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
        mi = torch.zeros(1)
        for el1_idx, pdf_x1_el in enumerate(pdf_x1):
            for el2_idx, pdf_x2_el in enumerate(pdf_x2):
                mi += pdf_x1x2[el1_idx, el2_idx] * torch.log2(
                    (pdf_x1x2[el1_idx, el2_idx] / (pdf_x1_el * pdf_x2_el)) + 1E-10)
        list_mi.append([mi.item(), epoch])
    return list_mi, net,list_spikecount


def sim_batch_final(dataset, device, neuron, varying_element, rank_NMF, model, list_mi=[], results_dir=None, net=None,
                    neuron_id=4, epoch=0):
    list_mi_local = []
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
        neuron.reset()
        s_out_rec = []
        for t in range(x_local.shape[1]):
            out = neuron(x_local[:, t, None, None])  # shape: n_batches x 1 fanout x n_param_values x n_channels
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
        # s_out_rec_train = torch.permute(s_out_rec_train, (0, 1, 3, 2))
        V_matrix_flattn = torch.flatten(s_out_rec_train, start_dim=2, end_dim=3)
        V_matrix = V_matrix_flattn[:, 0, :]

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


        if run_with_fake_input:
            # Store V input:
            with open(results_dir + 'W.pickle', 'wb') as f:
                pickle.dump(net.W.clone().detach().cpu()(), f)





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

    fig0,axis0 = plt.subplots(nrows = 1, ncols =1)
    axis0.imshow(V_matrix.clone().detach().cpu(),aspect='auto')
    axis0.set_xlabel('Time')
    axis0.set_ylabel('Trial x Stimulus')
    axis0.set_title('Raster Plot')

    fig1, axis1 = plt.subplots(nrows=1, ncols=1)

    im = axis1.imshow(pdf_x1x2.clone().detach().cpu(), aspect = 'auto')
    # plt.xticks([i for i in range(len(predicted_range))], np.array(predicted_range.cpu()))
    axis1.set_xticks([i for i in range(len(predicted_range))])
    axis1.set_xticklabels(np.array(predicted_range.cpu()))
    # plt.yticks([i for i in range(len(label_unique))], np.array(label_unique.cpu()))
    axis1.set_yticks([i for i in range(len(label_unique))])
    axis1.set_yticklabels(np.array(label_unique.cpu()))
    axis1.set_xlabel('Predicted')
    axis1.set_ylabel('Label')
    plt.colorbar(im, ax=axis1)
    if save_out:
        fig1.savefig(results_dir + 'pdf_joint.pdf', format='pdf')

    fig2, axis2 = plt.subplots(nrows=1, ncols=1)

    axis2.imshow(x_local[:, neuron_id, :].clone().detach().cpu(), aspect='auto')
    if save_out:
        fig2.savefig(results_dir + 'xlocal.pdf', format='pdf')

    fig3, axis3 = plt.subplots(nrows=1, ncols=1)
    axis3.imshow(H[neuron_id].clone().detach().cpu(), aspect='auto', interpolation='nearest')
    # plt.ylim([0, 128*2])
    axis3.set_xlabel('Time')
    axis3.set_ylabel('TrialxVariable')
    if save_out:
        fig3.savefig(results_dir + 'H.pdf', format='pdf')

    fig4, axis4 = plt.subplots(nrows=1, ncols=1)
    # plt.title('W NMF')
    axis4.imshow(net.W.clone().detach().cpu(), aspect='auto', interpolation='nearest')
    axis4.set_xlabel('Time')
    axis4.set_ylabel('Rank')
    if save_out:
        fig4.savefig(results_dir + 'W_nmf.pdf', format='pdf')

    fig5, axis5 = plt.subplots(nrows=1, ncols=1)
    im = axis5.imshow(torch.concat(output_vs_label, dim=1).clone().detach().cpu(), aspect='auto', cmap='seismic',
               interpolation='nearest')
    plt.colorbar(im,ax = axis5)
    # plt.title('OUTPUT vs LABEL')
    # plt.xlabel('Output|Label')
    axis5.set_ylabel('TrialxVariable')
    axis5.set_xticks([0, 1])
    axis5.set_xticklabels(['Output', 'Label'])
    if save_out:
        fig5.savefig(results_dir + 'Predicted_vs_label.pdf', format='pdf')

    fig6, axis6 = plt.subplots(nrows=1, ncols=1)
    axis6.imshow(net().clone().detach().cpu(), aspect='auto', interpolation='nearest')
    # plt.ylim([0, 128*2])
    axis6.set_xlabel('Time')
    axis6.set_ylabel('TrialxVariable')
    if save_out:
        fig6.savefig(results_dir + 'Out_nmf.pdf', format='pdf')


    result_plots = {'spikes':[fig0,axis0],'pdf':[fig1,axis1],'xlocal':[fig2,axis2],'H':[fig3,axis3],'W':[fig4,axis4],'output_vs_label':[fig5,axis5],'out_NMF':[fig6,axis6]}
    # list_mi.append(torch.mean(torch.Tensor(list_mi_local)))
    return list_mi, net,result_plots


def MI_neuron_params(neuron_param_values, name_param_sweeped, extremes_sweep, MNclass,letter = []):
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
        results_dir = set_results_folder([exp_id,str(letter),MNclass, name_param_sweeped, str(extremes_sweep)])
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
    dt = (1 / 100.0) / upsample_fac
    file_name = "data/data_braille_letters_all.pkl"
    if type(letter) != list:
        letter = [letter]
    data, labels, _, _, _, _ = load_data(file_name, upsample_fac, specify_letters=letter)
    nb_channels = data.shape[-1]

    x_train, x_test, y_train, y_test = train_test_split(
        data.cpu(), labels.cpu(), test_size=0.2, shuffle=True, stratify=labels.cpu()
    )

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)
    # ds_train, ds_test, labels, nb_channels, data_steps = load_analog_data(file_name, upsample_fac,
    #                                                                       specify_letters=['A'])
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



    encoder_weight_scale = 1.0

    ###########################################
    ##                Network                ##
    ###########################################

    # a = torch.empty((nb_inputs,))
    n_param_values = params['n_param_values']
    # a = torch.Tensor(np.linspace(-10, 10, n_param_values)).to(device)

    tensor_params = dict.fromkeys(neuron_param_values.keys(), None)
    for key in tensor_params.keys():
        tensor_params[key] = torch.Tensor(neuron_param_values[key]).to(device)

    varying_element = tensor_params[name_param_sweeped]

    # a = torch.Tensor(neuron_param_values['a']).to(device)
    # # nn.init.normal_(
    # #     a, mean=MNparams_dict['A2B'][0], std=fwd_weight_scale / np.sqrt(nb_inputs))
    #
    # A1 = torch.Tensor(neuron_param_values['A1']).to(device)
    #
    # A2 = torch.Tensor(neuron_param_values['A2']).to(device)

    fanout = 1  # number of output neurons from the linear expansion
    # TODO: Change input parameters list
    neuron = models.MN_neuron_IT(params['nb_channels'], fanout, n_param_values,
                                 tensor_params['a'],
                                 tensor_params['A1'],
                                 tensor_params['A2'],
                                 tensor_params['b'],
                                 tensor_params['G'],
                                 tensor_params['k1'],
                                 tensor_params['k2'],
                                 tensor_params['gain'],
                                 train=False, dt = dt)

    batch_size = int(params['batch_size'])

    inputDim = 1  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    learningRate = params['learningRate']  # 10#0.01

    neuron_id = int(params['neuron_id'])

    rank_NMF = int(params['rank_NMF'])
    range_weight_init = 10
    model = NlinearRegression(rank_NMF, 1)
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
    list_spikecount = {'mean':[],'std':[]}
    t_start = time.time()
    H_initial = None
    # net = NMF((), rank=rank_NMF, H=H_initial)
    net = None
    pbar = trange(nb_epochs)
    for e in pbar:
        local_loss = []
        H_train = []
        H_test = []
        accs = []  # accs: mean training accuracies for each batch
        # print('Epoch', e)
        list_loss, net = sim_batch_training(dl_train, device, neuron, varying_element, rank_NMF, model,
                                            {'optimizer': optimizer, 'criterion': criterion}, list_loss=list_loss,
                                            results_dir=results_dir, net=net, neuron_id=neuron_id, epoch=e)
        list_mi, net,list_spikecount = sim_batch_testing(dl_test, device, neuron, varying_element, rank_NMF, model, list_mi=list_mi,
                                         results_dir=results_dir, net=net, neuron_id=neuron_id, epoch=e, list_spikecount= list_spikecount)
        pbar.set_postfix_str("Mutual Information: " + str(np.round(list_mi[e][0], 2)) + ' bits. Loss: ' + str(
            np.round(list_loss[e][0], 2)) + ' Avg spike count: ' + str(np.round(list_spikecount['mean'][e])) + '(±' + str(np.round(list_spikecount['std'][e])) + ")")
        if np.isnan(list_loss[e][0]).all():
            print('All loss were nan, no spikes here. Exiting training')
            break
    train_duration = time.time() - t_start
    addToNetMetadata(metadatafilename, 'sim duration (sec)', train_duration)

    list_mi, net,result_plots = sim_batch_final(dl_test, device, neuron, varying_element, rank_NMF, model, list_mi=list_mi,
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
    with open(results_dir + 'spikecount.pickle', 'wb') as f:
        pickle.dump(list_spikecount,f)
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
    return result_plots,results_dir

if __name__ == "__main__":
    print('Current path',Current_PATH)
    writer = SummaryWriter(comment="MN_IT_concatenated.T")  # For logging purpose
    counter = 0
    letter = []
    letters = [letter]
    for letter in letters:
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
                    result_plots,results_dir = MI_neuron_params(dict_keys, name_param, variable_range_extremes, MNclass)
                    for key in result_plots:
                        result_plots[key][1].set_title('Class {}: Sweeping {} from {} to {}'.format(MNclass, name_param, variable_range_extremes[0],
                                                                        variable_range_extremes[1]))
                        writer.add_figure(key,result_plots[key][0],global_step=counter)
                    counter += 1
        sweep_range_string = [str(element) for element in sweep_ranges]
    plot_the_data(exp_id,sweep_range_string,results_dir,sweep_param_name,writer = writer,letters = letters)