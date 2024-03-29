import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from time import localtime, strftime
import matplotlib
import argparse
import seaborn as sns

matplotlib.pyplot.ioff()  # turn off interactive mode
import numpy as np
import pickle
import os
from training import MN_neuron
from utils_encoding import get_input_step_current, plot_outputs, pca_isi, plot_vmem, prepare_output_data, pca_timebins
from sklearn.covariance import MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from classifiers import MahalanobisClassifier
import seaborn as sns

sns.set_style('whitegrid')

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

    plot_outputs(dict_spk_rec, dict_mem_rec, list_mean_current, fig_folder=fig_folder)
    plot_vmem(dict_spk_rec, dict_mem_rec, list_mean_current, xlim=(0, 30), fig_folder=fig_folder)

    # PCA over time bins:
    # from dict_spk_rec of size (per class 'A' or 'C'):
    # 1 (single batch) x timebins x n_neurons=n_trials
    # to:
    # n_neurons=n_trials x PCs
    X_pca_timebins, Y_labels = pca_timebins(dict_spk_rec, class_labels, exp_variance=exp_variance,
                                            fig_folder=fig_folder)
    assert (X_pca_timebins.shape[0] == n_trials)
    assert (len(Y_labels) == n_trials)
    # X_pca_timebins.shape[0] = n trials (A + C)
    # X_pca_timebins.shape[1] = n features needed to explain exp_variance

    # Using Mahlanobis distance for bianry classification
    # Can we train a classifier to classify which neuron type was, based on the
    # neural activity?

    # Split dataset into test and train
    X_pca_timebins = pd.DataFrame(X_pca_timebins)
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(X_pca_timebins, Y_labels, test_size=test_size, random_state=42)
    assert (x_train.shape[0] == (1 - test_size) * X_pca_timebins.shape[0])

    # MahalanobisClassifier
    clf = MahalanobisClassifier(x_train, y_train)

    # Predicting
    pred_probs = clf.predict_probability(x_test)
    unique_labels = np.unique(Y_labels)
    pred_class = clf.predict_class(x_test, unique_labels)

    pred_actuals = pd.DataFrame([(pred, act) for pred, act in zip(pred_class, y_test)], columns=['pred', 'true'])
    truth = pred_actuals.loc[:, 'true']
    pred = pred_actuals.loc[:, 'pred']
    cm = confusion_matrix(truth, pred, labels=['A', 'C'])
    print('\nConfusion Matrix: \n', confusion_matrix(truth, pred))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=['A', 'C']
                                        )
    sns.set_style("white")
    cm_display.plot()
    fig = plt.gcf()
    fig.savefig(fig_folder.joinpath('Confusion Matrix.pdf'), format='pdf')
    plt.show()

    # Display input samples that were not classified correctly:
    sns.set_style("whitegrid")
    list_pairs = [('C', 'A'), ('C', 'C'), ('A', 'A')]
    dict_idx_true_predicted = dict.fromkeys(list_pairs)
    for pair in list_pairs:
        idx_true_predicted = np.where(np.logical_and(truth == pair[0], pred == pair[1]))[0]
        idx_true = inv_class_labels[pair[0]]
        idx_pred = inv_class_labels[pair[1]]
        assert (len(idx_true_predicted) == cm[idx_true, idx_pred])
        dict_idx_true_predicted[pair] = idx_true_predicted

    fig, axs = plt.subplots(3, 1)
    axs[0].set_title('True C - predicted A')
    axs[0].plot(torch.transpose(torch.tensor(x_train.values)[dict_idx_true_predicted[('C', 'A')], :], 0, 1), 'o-')
    axs[1].set_title('True C - predicted C')
    axs[1].plot(torch.transpose(torch.tensor(x_train.values)[dict_idx_true_predicted[('C', 'C')], :], 0, 1), 'o-')
    axs[2].set_title('True A - predicted A')
    axs[2].plot(torch.transpose(torch.tensor(x_train.values)[dict_idx_true_predicted[('A', 'A')], :], 0, 1), 'o-')
    fig.set_size_inches(20, 10)
    plt.subplots_adjust(hspace=0.3)
    fig.savefig(fig_folder.joinpath('Trials_classified.pdf'), format='pdf')
    plt.show()

    # Quantify how diagonal is the matrix:
    # Track the number of non diagonal elements, as a fraction of the total number of samples
    tot_trials = np.sum(cm)
    assert (tot_trials == (args.n_repetitions * args.nb_inputs * test_size * len(class_labels.keys())))
    output_data['confusion_matrix'] = cm
    output_data['spk_rec'] = {'A': dict_spk_rec['A'][0], 'C': dict_spk_rec['C'][0]}

    # ******************************************** Store data **********************************************************
    with open(exp_folder.joinpath('output_data.pickle'), 'wb') as f:
        pickle.dump(output_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('TODO')
    parser.add_argument('--MNclasses_to_test', type=list, default=['A', 'C'], help="learning rate")
    parser.add_argument('--nb_inputs', type=int, default=10)
    parser.add_argument('--n_repetitions', type=int, default=200)
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
