"""
Use this script to visualize the neuron traces according to the paper "A Generalized Linear Integrate-and-Fire Neural Model Produces Diverse Spiking Behaviors" 
by Stefan Mihalas and Ernst Niebur. Further, data was created with a fix length of 1sec (1ms time steps), with noise on the input current, 
and/or temporal jitter on the time point of the step for dynamic inputs. 

The script will also calculate the inter-spike intervalls (ISIs) for a single trial and for all repeating trials, whenever possible. 
For repeating trials, all ISIs are grouped and further statics represent the outcome of all repetitions per class.
"""


import pickle

from tqdm import tqdm

from tactile_encoding.utils.functions import (
    plot_isi_fix_len, plot_isi_fix_len_param_sweep, plot_isi_original,
    plot_single_isi_fix_len, plot_single_isi_fix_len_param_sweep,
    plot_traces_fix_len, plot_traces_fix_len_param_sweep, plot_traces_original)
from tactile_encoding.utils.utils import create_directory

path = './plots/original'  # set path to store plots
data_path = './data/original_mn_output'
create_directory(path)  # create folder if not existent
data_types = ['', '_noisy', '_temp_jitter', '_offset', '_noisy_temp_jitter',
              '_noisy_offset', '_temp_jitter_offset', '_noisy_temp_jitter_offset']

# data_types = ['_noisy_temp_jitter_offset']
max_trials = 1

if __name__ == '__main__':
    ###################
    # original length #
    ###################
    for counter, data_type in tqdm(enumerate(data_types)):
        # original
        if data_type == '':
            add_noise = False
            temp_jitter = False
            add_offset = False
        # single
        elif data_type == '_noisy':
            data_type = '_0.1_noise'
            add_noise = True
            temp_jitter = False
            add_offset = False
        elif data_type == '_temp_jitter':
            data_type = '_10_jitter'
            add_noise = False
            temp_jitter = True
            add_offset = False
        elif data_type == '_offset':
            data_type = '_0.1_offset'
            add_noise = False
            temp_jitter = False
            add_offset = True
        # combination of two
        elif data_type == '_noisy_temp_jitter':
            data_type = '_0.1_noise_10_jitter'
            add_noise = True
            temp_jitter = True
            add_offset = False
        elif data_type == '_noisy_offset':
            data_type = '_0.1_noise_0.1_offset'
            add_noise = True
            temp_jitter = False
            add_offset = True
        elif data_type == '_temp_jitter_offset':
            data_type = '_10_jitter_0.1_offset'
            add_noise = False
            temp_jitter = True
            add_offset = True
        # combination of three
        elif data_type == '_noisy_temp_jitter_offset':
            data_type = '_0.1_noise_10_jitter_0.1_offset'
            add_noise = True
            temp_jitter = True
            add_offset = True

        # load data
        filename = 'data_encoding_original' + data_type
        with open(f"{data_path}/{filename}.pkl", 'rb') as infile:
            data = pickle.load(infile)

        # create plots
        plot_traces_original(path, data, add_offset=add_offset, add_noise=add_noise,
                             temp_jitter=temp_jitter)

        plot_isi_original(path, data, add_offset=add_offset, add_noise=add_noise,
                          temp_jitter=temp_jitter, norm_count=True, norm_time=True)

    ##############
    # fix length #
    ##############
    for _, data_type in tqdm(enumerate(data_types)):
        # original
        if data_type == '':
            add_noise = False
            temp_jitter = False
            add_offset = False
        # single
        elif data_type == '_noisy':
            data_type = '_0.1_noise'
            add_noise = True
            temp_jitter = False
            add_offset = False
        elif data_type == '_temp_jitter':
            data_type = '_10_jitter'
            add_noise = False
            temp_jitter = True
            add_offset = False
        elif data_type == '_offset':
            data_type = '_0.1_offset'
            add_noise = False
            temp_jitter = False
            add_offset = True
        # combination of two
        elif data_type == '_noisy_temp_jitter':
            data_type = '_0.1_noise_10_jitter'
            add_noise = True
            temp_jitter = True
            add_offset = False
        elif data_type == '_noisy_offset':
            data_type = '_0.1_noise_0.1_offset'
            add_noise = True
            temp_jitter = False
            add_offset = True
        elif data_type == '_temp_jitter_offset':
            data_type = '_10_jitter_0.1_offset'
            add_noise = False
            temp_jitter = True
            add_offset = True
        # combination of three
        elif data_type == '_noisy_temp_jitter_offset':
            data_type = '_0.1_noise_10_jitter_0.1_offset'
            add_noise = True
            temp_jitter = True
            add_offset = True

        filename = 'data_encoding_fix_len' + data_type
        with open(f"{data_path}/{filename}.pkl", 'rb') as infile:
            data = pickle.load(infile)

        # create plots
        plot_traces_fix_len(path, data, max_trials=max_trials,
                            add_offset=add_offset, add_noise=add_noise, temp_jitter=temp_jitter)

        plot_single_isi_fix_len(path, data, max_trials=max_trials,
                                add_offset=add_offset, add_noise=add_noise, temp_jitter=temp_jitter, norm_count=True, norm_time=True)

        plot_isi_fix_len(path, data, max_trials=max_trials,
                         add_offset=add_offset, add_noise=add_noise, temp_jitter=temp_jitter, norm_count=True, norm_time=True)

    ##################
    # parameter weep #
    ##################
    # noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2, 5, 10]
    # offset_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2, 5, 10]
    # jitter = 10

    # add_noise = True
    # temp_jitter = True
    # add_offset = True

    # for offset_counter, offset in tqdm(enumerate(noise_levels)):
    #     for noise_counter, noise in tqdm(enumerate(noise_levels)):
    #         # load data
    #         data_type = f'{noise}_noise_{jitter}_jitter_{offset}_offset'
    #         filename = 'data_encoding_fix_len_' + data_type
    #         infile = open(f"{data_path}/{filename}.pkl", 'rb')
    #         data = pickle.load(infile)
    #         infile.close()

    #         # create plots
    #         # TODO include param values in figure name!!!
    #         plot_traces_fix_len_param_sweep(path, data, max_trials=max_trials, offset=offset, noise=noise, jitter=jitter,
    #                                         add_offset=add_offset, add_noise=add_noise, temp_jitter=temp_jitter)

    #         plot_single_isi_fix_len_param_sweep(path, data, max_trials=max_trials, offset=offset, noise=noise, jitter=jitter,
    #                                             add_offset=add_offset, add_noise=add_noise, temp_jitter=temp_jitter, norm_count=True, norm_time=True)

    #         plot_isi_fix_len_param_sweep(path, data, max_trials=max_trials, offset=offset, noise=noise, jitter=jitter,
    #                                      add_offset=add_offset, add_noise=add_noise, temp_jitter=temp_jitter, norm_count=True, norm_time=True)
