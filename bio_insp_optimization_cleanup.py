"""
The bio inspired optimization is based on the evolution scheme.
It goes as follows:
1. create intial population of size P
loop:
    2. validate fitness of single individual P_n
    3. select best x inidividuals and pertubate the genes (neuron parameters) 
    and include y random individuals to create population of P
    4. reached stop criterion end
"""

import os
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from datasets import load_data
from rsnn import *

# Settings for the SNN
global use_trainable_out
use_trainable_out = False
global use_trainable_tc
use_trainable_tc = False
global use_dropout
use_dropout = False
global batch_size
batch_size = 128
global lr
lr = 0.0001

# Init evolutionary algorithm
generations = 2  # number of generations to calculate
P = 5  # number of individuals in populations
# set the number of epochs you want to train the network (default = 300)
epochs = 2
save_fig = True  # set True to save the plots

letters = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# create folder to safe plots later (if not present)
if save_fig:
    path = '../plots'
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)

# check for available GPU and distribute work
if torch.cuda.device_count() > 1:
    torch.cuda.empty_cache()

    gpu_sel = 1
    gpu_av = [torch.cuda.is_available()
              for ii in range(torch.cuda.device_count())]
    print("Detected {} GPUs. The load will be shared.".format(
        torch.cuda.device_count()))
    for gpu in range(len(gpu_av)):
        if True in gpu_av:
            if gpu_av[gpu_sel]:
                device = torch.device("cuda:"+str(gpu))
                # torch.cuda.set_per_process_memory_fraction(0.9, device=device)
                print("Selected GPUs: {}" .format("cuda:"+str(gpu)))
            else:
                device = torch.device("cuda:"+str(gpu_av.index(True)))
        else:
            device = torch.device("cpu")
            print("No GPU detected. Running on CPU.")
else:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        print("Single GPU detected. Setting up the simulation there.")
        device = torch.device("cuda:0")
        # torch.cuda.set_per_process_memory_fraction(0.9, device=device)
    else:
        device = torch.device("cpu")
        print("No GPU detected. Running on CPU.")


# Init neuron model
neuron_model = 'mn_neuron'  # iz_neuon, lif_neuron

# Mihilas-Niebur neuron
if neuron_model == 'mn_neuron':
    from parameters.encoding_parameter import mn_parameter
    from models import MN_neuron
    neuron = MN_neuron
    parameter = mn_parameter
# Izhikevich neuron
elif neuron_model == 'iz_neuron':
    from parameters.encoding_parameter import iz_parameter
    from models import IZ_neuron
    neuron = IZ_neuron
    parameter = iz_parameter
# LIF neuron
else:
    from parameters.encoding_parameter import lif_parameter
    from models import LIF_neuron
    neuron = LIF_neuron
    parameter = lif_parameter

# Change list to include all the parameters to optimize
parameters_list = ["a", "A1", "A2"]  # a, A1, A2, b, G, k1, k2, R1, R2
parameter_to_optimize = []
for _, param in enumerate(parameter):
    if param[0] in parameters_list:
        parameter_to_optimize.append(param)

record = []
population_list = []
param_width = []
# create inital populataton of size P
for counter in range(P):
    individual = {}
    individual['identifier'] = counter
    # create inital parameter values
    for _, param in enumerate(parameter_to_optimize):
        # create parameter space to draw from
        # 2* increases the searchspace
        param_space = np.linspace(
            param[1]-0.5*abs(param[1]), param[2]+0.5*abs(param[2]), 100)
        # draw a random number out of parameter space
        individual[param[0]] = random.choice(param_space)
        # extract the parameter width for later
        if counter == 0:
            # define parameter width (min - max)
            # see above (2*)
            param_width.append(
                np.diff((param[1]-0.5*abs(param[1]), param[2]+0.5*abs(param[2]))))
    population_list.append(individual)

sampling_freq = 100.0  # Hz
upsample_fac = 1.0  # 10.0
frequ = sampling_freq * upsample_fac
dt = 1/frequ

# preprocess data
print("Start preparing data.")
data_neuron, labels, timestamps, data_steps, labels_as_number, data = load_data(
    "./data/data_braille_letters_all.pkl", upsample_fac=upsample_fac, norm_val=2, filtering=True)
print("Finished data prepartion.\n")
# linear decrease


def calc_sigma(sigma_start, sigma_stop, generations, x):
    sigma = ((sigma_stop-sigma_start)/generations)*x+sigma_start
    return sigma


# init datastorage
print("Initialize data storage.")
file_storage_found = False
idx_file_storage = 1
while not file_storage_found:
    file_storage_path = './results/record_' + str(idx_file_storage) + '.pkl'
    if os.path.isfile(file_storage_path):
        idx_file_storage += 1
    else:
        file_storage_found = True

# create folder to safe plots later (if not present)
if save_fig:
    path_for_plots = './plots/record_' + str(idx_file_storage)
    isExist_record = os.path.exists(path_for_plots)

    if not isExist_record:
        os.makedirs(path_for_plots)
print("Data storage initialized.\n")

print("Starting optimization.")
# TODO inlcude a first split to have the validation set
# TODO define another end criterion (saturation in accuracy for n runs?)
# iterate over generataions
for generation in range(generations):
    highest_fitness = 0.0
    best_individual = []
    very_best_layers = []

    # set seed for train-test split (fix seed for one generation to allow comparability)
    global seed
    seed = random.randint(0, 2**32 - 1)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed set to {}".format(seed))

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # find fitness for each individual
    for identifier, individual in enumerate(population_list):
        print("\nStarting individual {} of {} in generation {} of {}.".format(
            identifier+1, len(population_list), generation+1, generations))
        # create neuron response for individual
        neurons = MN_neuron(24, individual, dt=dt, train=False)

        input = data_neuron
        output_s = []

        for t in range(input.shape[0]):
            out = neurons(input[t])
            output_s.append(out.cpu().numpy())
        output_s = np.stack(output_s)

        # split in train-test set
        output_s = torch.as_tensor(output_s, dtype=torch.float)
        x_train, x_test, y_train, y_test = train_test_split(
            output_s, labels_as_number, test_size=0.20, shuffle=True, stratify=labels_as_number, random_state=seed)
        ds_train = TensorDataset(x_train, y_train)
        ds_test = TensorDataset(x_test, y_test)

        # calculate fitness
        # initialize and train network
        loss_hist, acc_hist, best_layers = build_and_train(
            device, data_steps, dt, ds_train, ds_test, epochs,
            labels_as_number, lr, seed, use_trainable_tc=use_trainable_tc,
            use_trainable_out=use_trainable_out, 
            use_dropout=use_dropout)

        individual['fitness'] = max(acc_hist[1])*100
        if max(acc_hist[1]) > highest_fitness:
            # TODO inlcude std of acc here as second metric
            highest_fitness = max(acc_hist[1])
            best_individual = identifier
            very_best_layer = best_layers
        elif max(acc_hist[1]) == highest_fitness:
            # TODO use spike count second metric
            print("Find a second metric to deside which is better")

    # best individual
    print("*******************************************")
    print("Best individual: {}" .format(best_individual+1))
    print("*******************************************")

    # TODO do not keep all data in memory, but just load, append, and dump
    # save record for postprocessing
    record.append(population_list) # check if slicing ([:]) is still needed
    record.append(best_individual)
    # TODO create pandas df to dump
    with open(file_storage_path, 'wb') as f:
        pickle.dump(record, f)

    # TODO change to validation set here
    # plots of the best individual in this generation
    if save_fig:
        ConfusionMatrix(device, ds_test, save_fig, very_best_layer,
            generation, best_individual, letters, seed, seed_worker, batch_size, 
            use_trainable_tc=use_trainable_tc, use_trainable_out=use_trainable_out, 
            use_dropout=use_dropout)
        NetworkActivity(device, ds_test, save_fig, very_best_layer,
            generation, best_individual, seed, seed_worker, batch_size, 
            use_trainable_tc=use_trainable_tc, use_trainable_out=use_trainable_out, 
            use_dropout=use_dropout)

    # do not create a new generation in the last trial
    if generation < generations-1:
        # calc sigma to reduce searchspace over generations
        # start at 100% and end at 1% of search space
        sigma = calc_sigma(1.0, 0.01, generations, generation)

        # create next generation
        best_individual_dict = population_list[best_individual]
        population_list = []
        for counter in range(P):
            individual = {}
            individual['individual'] = counter+1
            # keep best found individual so far (no perturbation)
            if counter == 0:
                for _, param in enumerate(parameter_to_optimize):
                    individual[param[0]] = best_individual_dict[param[0]]
            else:
                # create first 75% from best
                if counter <= 0.75*P:
                    for counter, param in enumerate(parameter_to_optimize):
                        # (mu, sigma, nb_samples)
                        new_val = np.random.normal(
                            best_individual_dict[param[0]], param_width[counter]*sigma, 1)
                        individual[param[0]] = new_val[0]
                # create remaining 25% random
                else:
                    for _, param in enumerate(parameter_to_optimize):
                        # create parameter space to draw from
                        param_space = np.linspace(
                            param[1]-0.5*abs(param[1]), param[2]+0.5*abs(param[2]), 100)
                        # draw a random number out of parameter space
                        individual[param[0]] = random.choice(param_space)
            population_list.append(individual)

    print("Finished generation {} of {}.".format(generation+1, generations))
    print("###################################################\n\n")

print("End of the evolution reached")