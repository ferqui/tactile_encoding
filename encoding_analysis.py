import pandas as pd
import torch
import matplotlib.pyplot as plt
from time import localtime, strftime
import matplotlib
import argparse
import seaborn as sns
matplotlib.pyplot.ioff()  # turn off interactive mode
import numpy as np
import os
from training import MN_neuron
from utils_encoding import get_input_step_current, plot_outputs, pca_isi

Current_PATH = os.getcwd()
torch.manual_seed(0)

MNclass_to_param = {
    'A': {'a': 0, 'A1': 0, 'A2': 0},
    'C': {'a': 5, 'A1': 0, 'A2': 0}
}
class_labels = dict(zip(list(np.arange(len(MNclass_to_param.keys()))), MNclass_to_param.keys()))
inv_class_labels = {v: k for k, v in class_labels.items()}

def main(args):
    #exp_id = strftime("%d%b%Y_%H-%M-%S", localtime())

    list_classes = args.MNclasses_to_test
    nb_inputs = args.nb_inputs
    amplitudes = np.arange(1, nb_inputs + 1)
    n_repetitions = args.n_repetitions
    sigma = args.sigma

    # each neuron receives a different input amplitude
    dict_spk_rec = dict.fromkeys(list_classes, [])
    dict_mem_rec = dict.fromkeys(list_classes, [])
    for MN_class_type in list_classes:
        neurons = MN_neuron(len(amplitudes)*n_repetitions, MNclass_to_param[MN_class_type], dt=args.dt, train=False)

        x_local = get_input_step_current(dt_sec=args.dt, stim_length_sec=args.stim_length_sec, amplitudes=amplitudes,
                                         n_trials=n_repetitions, sig=sigma)

        neurons.reset()
        spk_rec = []
        mem_rec = []
        for t in range(x_local.shape[1]):
            out = neurons(x_local[:, t])

            spk_rec.append(neurons.state.spk)
            mem_rec.append(neurons.state.V)

        dict_spk_rec[MN_class_type] = torch.stack(spk_rec, dim=1) # shape: batch_size, time_steps, neurons (i.e., current amplitudes)
        dict_mem_rec[MN_class_type] = torch.stack(mem_rec, dim=1)

    fig = plot_outputs(dict_spk_rec, dict_mem_rec, dt=args.dt)
    fig.savefig('Output_MN.pdf', format='pdf')

    X_pca_isi = pca_isi(dict_spk_rec, class_labels)

    print('End')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('TODO')
    parser.add_argument('--MNclasses_to_test', type=list, default=['A', 'C'], help="learning rate")
    parser.add_argument('--nb_inputs', type=int, default=10)
    parser.add_argument('--n_repetitions', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=0, help='sigma gaussian distrubution of I current')
    parser.add_argument('--stim_length_sec', type=float, default=0.2)
    parser.add_argument('--selected_input_channel', type=int, default=0)
    parser.add_argument('--dt', type=float, default=0.001)

    args = parser.parse_args()

    main(args)
