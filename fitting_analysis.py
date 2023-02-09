import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from time import localtime, strftime
import matplotlib
import argparse
import seaborn as sns
import scipy.integrate as scint
matplotlib.pyplot.ioff()  # turn off interactive mode
import numpy as np
import pickle
import os
from training import MN_neuron
from utils_encoding import get_input_step_current, plot_outputs, pca_isi, plot_vmem, prepare_output_data, pca_timebins
from sklearn.covariance import MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from classifiers import MahalanobisClassifier
from scipy.optimize import curve_fit
#
# def func(x  ,t, a, b, c,d):
#     return a*np.exp(b*x) + c/np.log(d*x)

y0 = 0


# def func(time,b,c,x_local_full,e):
#
#     return scint.odeint(pend, y0, time, args=(b, c, x_local_full))[1:,0]
# def func_ode(t,a,b,c,d):
#     tspan = np.hstack([[0],np.hstack([t])])
#     return scint.odeint(func,0, tspan, args=(a,b,c,d))[1:,0]

# sol = scint.odeint(pend, y0, t, args=(b, c))


# popt, pcov = curve_fit(f, t, sol[:,0], p0=guess)

torch.manual_seed(0)
np.random.seed(19)
##########################################################
# Settings/Path:
Current_PATH = os.getcwd()
output_folder = Path('./results')
output_folder.mkdir(parents=True, exist_ok=True)

MNclass_to_param = {
    'A': {'a': 0, 'A1': 0, 'A2': 0},
    'C': {'a': 5, 'A1': 0, 'A2': 0}
}
class_labels = dict(zip(list(np.arange(len(MNclass_to_param.keys()))),
                        MNclass_to_param.keys()))
inv_class_labels = {v: k for k, v in class_labels.items()}

class fitClass:

    def __init__(self,stim):
        self.stim = stim
        self.b = 1
        self.c = 1
        pass
    def pend(self,y, t, b, c):
        dydt = -b * np.exp(c*y) + self.stim[int(t)]
        return dydt
    def func(self,time,b,c):
        # print(time)

        tspan = np.hstack([[0], np.hstack([time])])

        return scint.odeint(self.pend, y0, tspan, args=(b, c))[1:,0]
############################################################

def main(args):
    # Prepare path:
    exp_id = strftime("%d%b%Y_%H-%M-%S", localtime())

    exp_folder = output_folder.joinpath(exp_id)
    exp_folder.mkdir(parents=True, exist_ok=True)

    fig_folder = exp_folder.joinpath('figures')
    fig_folder.mkdir(parents=True, exist_ok=True)

    output_data = prepare_output_data(args)

    # Input arguments:
    list_classes = args.MNclasses_to_test
    nb_inputs = args.nb_inputs
    # Linearly map the number of inputs to a range of input current amplitudes.
    amplitudes = np.arange(1, nb_inputs + 1) * args.gain + args.offset
    n_repetitions = args.n_repetitions
    sigma = args.sigma
    n_trials = args.n_repetitions * args.nb_inputs * len(list_classes)
    exp_variance = args.exp_variance

    # each neuron receives a different input amplitude
    dict_spk_rec = dict.fromkeys(list_classes, [])
    dict_mem_rec = dict.fromkeys(list_classes, [])
    for MN_class_type in list_classes:
        x_local_full = []
        neurons = MN_neuron(len(amplitudes) * n_repetitions, MNclass_to_param[MN_class_type], dt=args.dt, train=False)

        x_local, list_mean_current = get_input_step_current(dt_sec=args.dt, stim_length_sec=args.stim_length_sec,
                                                            amplitudes=amplitudes,
                                                            n_trials=n_repetitions, sig=sigma)

        neurons.reset()
        spk_rec = []
        mem_rec = []
        for t in range(x_local.shape[1]):
            out = neurons(x_local[:, t])

            spk_rec.append(neurons.state.spk)
            mem_rec.append(neurons.state.V)

        dict_spk_rec[MN_class_type] = torch.stack(spk_rec,
                                                  dim=1)  # shape: batch_size, time_steps, neurons (i.e., current amplitudes)
        dict_mem_rec[MN_class_type] = torch.stack(mem_rec, dim=1)

        spks = torch.where(dict_spk_rec[MN_class_type] == 1)
        # plt.imshow(dict_spk_rec['A'][0,:,:],aspect='auto')
        x_local_sampled_list = []
        isi_tensor_list = []
        time_list = []
        x_local_sampled = torch.zeros_like(x_local)
        x_local_full = x_local.flatten()
        isi_tensor = torch.zeros_like(x_local)
        fig1, axis1 = plt.subplots(nrows=2, ncols=1,sharex=True)
        eee = plt.get_cmap('inferno')
        uuu = eee(np.linspace(0, 1, 10))
        time_last = 0
        for i in range(dict_spk_rec[MN_class_type].shape[2]):
            idx = torch.where(spks[2] == i)
            time = spks[1][idx]
            time_list.append(time+time_last)
            x_local_sampled_list.append(x_local[0,time,i])
            isi = torch.diff(time)
            isi_tensor_list.append(torch.cat([torch.tensor([isi[0]]).to(torch.float64), isi.to(torch.float64)]))
            time_last += time[-1]
            axis1[0].plot([i for i in range(x_local.shape[1])], x_local[0, :, i], '-D', markevery=list(time),color = uuu[int(i/n_repetitions)])
            # axis1[0].plot(time,x_local[0,time,0],'.')
            # axis1.eventplot(time,lineoffsets=1.4)
            axis1[1].plot(time, torch.cat([torch.tensor([isi[0]]).to(torch.float64), isi.to(torch.float64)]), 'x',color = uuu[int(i/n_repetitions)])
            # isi_tensor[0,time,i] = torch.cat([torch.tensor([0]),isi.to(torch.float64)])
            # plt.plot(time[:-1],isi)
            fig1.suptitle('Class ' + MN_class_type)
            axis1[0].set_title('Stimulus')
            axis1[1].set_xlabel('Time')
            axis1[1].set_title('ISI')
            axis1[0].set_ylabel('Current (A)')
            axis1[1].set_ylabel('ISI')
        plt.figure()
        time_tensor = torch.concat(time_list)
        x_local_sampled = torch.concat(x_local_sampled_list)
        isi_tensor = torch.concat(isi_tensor_list)
        b = 1
        c = 1
        d = x_local_sampled.numpy()
        e = time_tensor
        time = np.linspace(0,len(x_local_full)-1,len(x_local_full))
        padding = np.zeros_like(x_local_full)
        padding[time_tensor] = isi_tensor
        myfit = fitClass(x_local_full.numpy())
        popt, pcov = curve_fit(myfit.func, time, padding)
        print('we')
        # popt, pcov = curve_fit(f,y0, sol[], isi_tensor)
        # plt.plot(time_list, func(x_local_sampled, *popt), 'r',label='fit')
        # plt.plot(time_list,x_local_sampled,'b',label='input')
        # label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        # plt.plot(time_list, isi_tensor,'g',label='data')
        plt.title('Class ' + MN_class_type)
        plt.xlabel('Time * Stimulus')
        plt.ylabel('ISI (ms)')
        plt.plot(time_list[0], x_local_sampled_list[0], 'b+-', label='input')
        plt.plot(time_list[0], isi_tensor_list[0], 'r*-', label='data')
        #mystr = str(np.round(popt[0])) + "exp(" + str(np.round(popt[1],2)) + "x)  + "+str(np.round(popt[2],2)) + "*1/ln("+str(np.round(popt[4],2)) + ")"
        mystr = ''
        # plt.plot(time_list[0], func(x_local_sampled_list[0], *popt), 'gx-', label=mystr)
        plt.plot(time, myfit.func(time, *popt), 'gx-')

        for i in range(len(isi_tensor_list))[1:]:

            plt.plot(time_list[i], x_local_sampled_list[i], 'b+-')
            plt.plot(time_list[i], isi_tensor_list[i], 'r*-')
        plt.legend()
        # plt.legend()
    plt.show()

    print('Hello')
    # ******************************************** Store data **********************************************************
    with open(exp_folder.joinpath('output_data.pickle'), 'wb') as f:
        pickle.dump(output_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('TODO')
    parser.add_argument('--MNclasses_to_test', type=list, default=['A', 'C'], help="learning rate")
    parser.add_argument('--nb_inputs', type=int, default=10)
    parser.add_argument('--n_repetitions', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=0, help='sigma gaussian distribution of I current')
    # NOTE: The number of input neurons = number of different input current amplitudes
    parser.add_argument('--gain', type=int, default=1)
    parser.add_argument('--offset', type=int, default=1)
    parser.add_argument('--stim_length_sec', type=float, default=0.2)
    parser.add_argument('--selected_input_channel', type=int, default=0)
    parser.add_argument('--exp_variance', default=.95)
    parser.add_argument('--dt', type=float, default=0.001)

    args = parser.parse_args()

    main(args)
