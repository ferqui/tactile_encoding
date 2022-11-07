import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import trange

import numpy as np

from datasets import load_analog_data
from parameters.MN_params import MNparams_dict, INIT_MODE
from models import Encoder, LIF_neuron, MN_neuron_sp

from auxiliary import compute_classification_accuracy, plot_spikes

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

firing_mode_dict = {
    'FA':{'a': 5, 'A1': 0, 'A2': 0},
    'SA':{'a': 0, 'A1': 0, 'A2': 0},
    'MIX':{'a': 5, 'A1': 5, 'A2': -0.3},
}

def main(args):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cpu'
    ###########################################
    ##                Dataset                ##
    ###########################################
    upsample_fac = 1
    file_name = "data/data_braille_letters_digits.pkl"
    ds_train, ds_test, labels, nb_channels, data_steps = load_analog_data(file_name, upsample_fac,
                                                                          specify_letters=[])


    # Network parameters
    nb_input_copies = args.expansion
    nb_inputs = nb_channels * nb_input_copies
    nb_hidden = 450
    nb_outputs = len(np.unique(labels))

    # Learning parameters
    nb_epochs = 300

    # Neuron parameters
    tau_mem = args.tau_mem  # ms
    tau_syn = tau_mem / args.tau_ratio
    alpha = float(np.exp(-0.001 / tau_syn))
    beta = float(np.exp(-0.001 / tau_mem))

    encoder_weight_scale = 1.0
    fwd_weight_scale = args.fwd_weight_scale
    rec_weight_scale = args.weight_scale_factor * fwd_weight_scale

    ###########################################
    ##                Network                ##
    ###########################################

    # a = torch.empty((nb_inputs,))
    # nn.init.normal_(
    #     a, mean=MNparams_dict[INIT_MODE][0], std=fwd_weight_scale / np.sqrt(nb_inputs))

    # A1 = torch.empty((nb_inputs,))
    # nn.init.normal_(
    #     A1, mean=MNparams_dict[INIT_MODE][1], std=fwd_weight_scale / np.sqrt(nb_inputs))

    # A2 = torch.empty((nb_inputs,))
    # nn.init.normal_(
    #     A2, mean=MNparams_dict[INIT_MODE][2], std=fwd_weight_scale / np.sqrt(nb_inputs))
    a = 5
    A1 = 10
    A2 = 1
    b = 10
    G = 50
    k1 = 200
    k2 = 20
    R1 = 0
    R2 = 1
    a = nn.Parameter(torch.Tensor([a]), requires_grad = True).to(device)
    A1 = nn.Parameter(torch.Tensor([A1]), requires_grad = True).to(device)
    A2 = nn.Parameter(torch.Tensor([A2]), requires_grad = True).to(device)
    b = nn.Parameter(torch.Tensor([b]), requires_grad = True).to(device)
    G = nn.Parameter(torch.Tensor([G]), requires_grad = False).to(device)
    k1 = nn.Parameter(torch.Tensor([k1]), requires_grad = False).to(device)
    k2 = nn.Parameter(torch.Tensor([k2]), requires_grad = False).to(device)
    R1 = nn.Parameter(torch.Tensor([R1]), requires_grad = False).to(device)
    R2 = nn.Parameter(torch.Tensor([R2]), requires_grad = False).to(device)
    torch.autograd.set_detect_anomaly(True)
    network = nn.Sequential(Encoder(nb_inputs, encoder_weight_scale, nb_input_copies),
                            MN_neuron_sp(nb_inputs, firing_mode_dict[args.firing_mode], train=args.train, a = a, A1 = A1, A2 = A2,  b=b, G=G, k1=k1, k2=k2, R1=R1, R2=R2),
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

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    pbar = trange(nb_epochs)
    parameter_rec = {'a': [], 'A1':[], 'A2':[],'w1':[],'w1_rec':[],'w2':[]}

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
            for t in range(x_local.shape[1]):
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
            reg_loss = args.reg_spikes * torch.mean(
                torch.sum(spk_rec, 1))  # e.g., L1 loss on total number of spikes (original: 1e-3)
            reg_loss += args.reg_neurons * torch.mean(
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
        parameter_rec['a'].append(a.clone().detach().numpy())
        parameter_rec['A1'].append(A1.clone().detach().numpy())
        parameter_rec['A2'].append(A2.clone().detach().numpy())

        parameter_rec['w1'].append(network[-2].weight.clone().detach().numpy())
        parameter_rec['w1_rec'].append(network[-2].weight_rec.clone().detach().numpy())
        parameter_rec['w2'].append(network[-1].weight.clone().detach().numpy())

        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)

        # mean_accs: mean training accuracy of current epoch (average over all batches)
        mean_accs = np.mean(accs)
        accs_hist[0].append(mean_accs)

        # Calculate test accuracy in each epoch on the testing dataset
        test_acc, test_ttc, spk_hidden, spk_output = compute_classification_accuracy(
            dl_test, network, True, device)
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

    plt.figure()
    plt.plot(np.array(parameter_rec['a']))
    plt.figure()
    plt.plot(np.array(parameter_rec['A1']))
    plt.figure()
    plt.plot(np.array(parameter_rec['A2']))
    plt.figure()
    plt.plot(np.array(parameter_rec['w1'])[:,:,0])
    plt.figure()
    plt.plot(np.array(parameter_rec['w1_rec'])[:,:,0])
    plt.figure()
    plt.plot(np.array(parameter_rec['w2'])[:,:,0])

    plt.show()
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Encoding')
    parser.add_argument('--firing-mode', type=str, default='FA', choices=['FA', 'SA', 'MIX'], 
        help="Choose between different firing modes")
    parser.add_argument('--norm', type=float, default=10.0,
        help='Data normalization')
    parser.add_argument('--expansion', type=int, default=1,
        help='Number of channel expansion (default: 1 (no expansion)).')
    parser.add_argument('--tau_mem', type=float, default=0.02,
        help='Membrane time constant.')
    parser.add_argument('--tau_ratio', type=float, default=2,
        help='Tau ratio.')
    parser.add_argument('--fwd_weight_scale', type=float, default=1,
        help='fwd_weight_scale.')
    parser.add_argument('--weight_scale_factor', type=float, default=0.01,
        help='weight_scale_factor')
    parser.add_argument('--reg_spikes', type=float, default=0.004,
        help='reg_spikes')
    parser.add_argument('--reg_neurons', type=float, default=0.000001,
        help='reg_neurons')

    parser.add_argument('--train', action="store_true",
        help='Train the MN neuron.')
    args = parser.parse_args()
    assert args.expansion > 0, "Expansion number should be greater that 0"
    main(args)