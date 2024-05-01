"""
Use this script to visualize the neuron traces according to the paper "A Generalized Linear Integrate-and-Fire Neural Model Produces Diverse Spiking Behaviors" 
by Stefan Mihalas and Ernst Niebur. Further, data was created with a fix length of 1sec (1ms time steps), with noise on the input current, 
and/or temporal jitter on the time point of the step for dynamic inputs. 

The script will also calculate the inter-spike intervalls (ISIs) for a single trial and for all repeating trials, whenever possible. 
For repeating trials, all ISIs are grouped and further statics represent the outcome of all repetitions per class.
"""


import pickle

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.metrics import silhouette_score
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

OFFSET = 0.1
NOISE = 0.2
JITTER = 10
NB_TRIALS = 3

def create_interactive_3d_plot(fig_path, title, x, y, z, objectives):
    # Create an interactive 3d surface plot
    plotly_fig = go.Figure(data=[
        go.Surface(x=x,
                   y=y,
                   z=z)
    ])

    # Update layout
    plotly_fig.update_layout(title=title,
                             scene=dict(xaxis_title=f'{objectives[0]}',
                                        yaxis_title=f'{objectives[1]}',
                                        zaxis_title='Silhouette score'))  # TODO check naming

    # Save the plot as an interactive HTML file
    plotly_fig.write_html(f"{fig_path}.html")

if __name__ == '__main__':
    ###################
    # original length #
    ###################
    # for counter, data_type in tqdm(enumerate(data_types)):
    #     # original
    #     if data_type == '':
    #         add_noise = False
    #         temp_jitter = False
    #         add_offset = False
    #     # single
    #     elif data_type == '_noisy':
    #         data_type = f'_{NOISE}_noise'
    #         add_noise = True
    #         temp_jitter = False
    #         add_offset = False
    #     elif data_type == '_temp_jitter':
    #         data_type = f'_{JITTER}_jitter'
    #         add_noise = False
    #         temp_jitter = True
    #         add_offset = False
    #     elif data_type == '_offset':
    #         data_type = f'_{OFFSET}_offset'
    #         add_noise = False
    #         temp_jitter = False
    #         add_offset = True
    #     # combination of two
    #     elif data_type == '_noisy_temp_jitter':
    #         data_type = f'_{NOISE}_noise_{JITTER}_jitter'
    #         add_noise = True
    #         temp_jitter = True
    #         add_offset = False
    #     elif data_type == '_noisy_offset':
    #         data_type = f'_{NOISE}_noise_{OFFSET}_offset'
    #         add_noise = True
    #         temp_jitter = False
    #         add_offset = True
    #     elif data_type == '_temp_jitter_offset':
    #         data_type = f'_{JITTER}_jitter_{OFFSET}_offset'
    #         add_noise = False
    #         temp_jitter = True
    #         add_offset = True
    #     # combination of three
    #     elif data_type == '_noisy_temp_jitter_offset':
    #         data_type = f'_{NOISE}_noise_{JITTER}_jitter_{OFFSET}_offset'
    #         add_noise = True
    #         temp_jitter = True
    #         add_offset = True

    #     # load data
    #     filename = 'data_encoding_original' + data_type
    #     with open(f"{data_path}/{filename}.pkl", 'rb') as infile:
    #         data = pickle.load(infile)

    #     # create plots
    #     plot_traces_original(path, data, add_offset=add_offset, add_noise=add_noise,
    #                          temp_jitter=temp_jitter)

    #     plot_isi_original(path, data, add_offset=add_offset, add_noise=add_noise,
    #                       temp_jitter=temp_jitter, norm_count=True, norm_time=True)

    # # ##############
    # # # fix length #
    # # ##############
    # for _, data_type in tqdm(enumerate(data_types)):
    #     # original
    #     if data_type == '':
    #         add_noise = False
    #         temp_jitter = False
    #         add_offset = False
    #     # single
    #     elif data_type == '_noisy':
    #         data_type = f'_{NOISE}_noise'
    #         add_noise = True
    #         temp_jitter = False
    #         add_offset = False
    #     elif data_type == '_temp_jitter':
    #         data_type = f'_{JITTER}_jitter'
    #         add_noise = False
    #         temp_jitter = True
    #         add_offset = False
    #     elif data_type == '_offset':
    #         data_type = f'_{OFFSET}_offset'
    #         add_noise = False
    #         temp_jitter = False
    #         add_offset = True
    #     # combination of two
    #     elif data_type == '_noisy_temp_jitter':
    #         data_type = f'_{NOISE}_noise_{JITTER}_jitter'
    #         add_noise = True
    #         temp_jitter = True
    #         add_offset = False
    #     elif data_type == '_noisy_offset':
    #         data_type = f'_{NOISE}_noise_{OFFSET}_offset'
    #         add_noise = True
    #         temp_jitter = False
    #         add_offset = True
    #     elif data_type == '_temp_jitter_offset':
    #         data_type = f'_{JITTER}_jitter_{OFFSET}_offset'
    #         add_noise = False
    #         temp_jitter = True
    #         add_offset = True
    #     # combination of three
    #     elif data_type == '_noisy_temp_jitter_offset':
    #         data_type = f'_{NOISE}_noise_{JITTER}_jitter_{OFFSET}_offset'
    #         add_noise = True
    #         temp_jitter = True
    #         add_offset = True

    #     filename = 'data_encoding_fix_len' + data_type
    #     with open(f"{data_path}/{filename}.pkl", 'rb') as infile:
    #         data = pickle.load(infile)

    #     # create plots
    #     plot_traces_fix_len(path, data, max_trials=NB_TRIALS,
    #                         add_offset=add_offset, add_noise=add_noise, temp_jitter=temp_jitter)

    #     plot_single_isi_fix_len(path, data, max_trials=NB_TRIALS,
    #                             add_offset=add_offset, add_noise=add_noise, temp_jitter=temp_jitter, norm_count=True, norm_time=True)

    #     plot_isi_fix_len(path, data, max_trials=NB_TRIALS,
    #                      add_offset=add_offset, add_noise=add_noise, temp_jitter=temp_jitter, norm_count=True, norm_time=True)

    ##################
    # parameter weep #
    ##################
    step_size = 0.05
    noise_levels = np.round(np.arange(0, 1+step_size, step_size), 5)
    offset_levels = noise_levels

    add_noise = True
    temp_jitter = True
    add_offset = True

    silhouette_sc_list = []

    for offset_counter, offset in tqdm(enumerate(offset_levels), position=0, leave=False, total=len(offset_levels)):
        silhouette_sc_list_offset = []
        for noise_counter, noise in tqdm(enumerate(noise_levels), position=1, leave=False, total=len(noise_levels)):
            # load data
            data_type = f'{noise}_noise_{JITTER}_jitter_{offset}_offset'
            filename = 'data_encoding_fix_len_' + data_type
            with open(f"{data_path}/{filename}.pkl", 'rb') as infile:
                data = pickle.load(infile)

            # create plots
            # plot_traces_fix_len_param_sweep(path, data, max_trials=NB_TRIALS, offset=offset, noise=noise, jitter=JITTER,
            #                                 add_offset=add_offset, add_noise=add_noise, temp_jitter=temp_jitter)

            # plot_single_isi_fix_len_param_sweep(path, data, max_trials=NB_TRIALS, offset=offset, noise=noise, jitter=JITTER,
            #                                     add_offset=add_offset, add_noise=add_noise, temp_jitter=temp_jitter, norm_count=True, norm_time=True)

            isis_list = plot_isi_fix_len_param_sweep(path, data, max_trials=NB_TRIALS, offset=offset, noise=noise, jitter=JITTER,
                                         add_offset=add_offset, add_noise=add_noise, temp_jitter=temp_jitter, norm_count=True, norm_time=True)
            # isis_list: (id, isi, count)*classes
            
            # each entry can have different length so we want to turn the data into one-hot encoding
            precision = 2  # in digits
            id = []
            unique_isis = []
            for entry in isis_list:
                id.append(entry[0])
                unique_isis.extend(np.unique(np.round(entry[1], precision)))
            unique_isis = np.unique(unique_isis)  
            
            # with that we found all ISIs in the dataset and now create a datastructure of size trials*unique_isis
            data = np.zeros((len(id), len(unique_isis)))
            for i, entry in enumerate(isis_list):
                if isinstance(entry[1], (int, float)):  # Check if entry[1] is a single element
                    data[i, np.where(unique_isis == entry[1])] = entry[2]  # Assign count directly
                else:
                    for j, isi in enumerate(entry[1]):
                        data[i, np.where(unique_isis == isi)] = entry[2][j]
            # get the silouhette score to qunatify how seperable the groups are
            silhouette_sc = silhouette_score(data, np.array(id))
            silhouette_sc_list_offset.append(silhouette_sc)
        silhouette_sc_list.append(silhouette_sc_list_offset)

    # create 3d plot
    fig_path = f'{path}/siouhette_score_parameter_sweep'
    title = 'Silhouette score depending on noise'
    objectives=['offset', 'noise']  # TODO double check order! (just reduce on in len or change numbers)
    create_interactive_3d_plot(fig_path=fig_path, title=title, x=offset_levels, y=noise_levels, z=silhouette_sc_list, objectives=objectives)
