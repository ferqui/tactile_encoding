import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from tactile_encoding.parameters.ideal_params import (input_currents,
                                                      neuron_parameters,
                                                      runtime, time_points)
from tactile_encoding.utils.models import MN_neuron
from tactile_encoding.utils.utils import value2key

SUPTITLE_FONT_SIZE = 16
TITLE_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 10


# TODO check if check_cuda() and running on GPU can speed things up


def original(offset=1E-1, noise=1E-1, jitter=10, add_offset=False, add_noise=False, add_jitter=False):
    """
    Recreates the original data for behavior classes 
    regarding neuron input current (profiles) over time 
    and neuron parameters. For different classes the 
    length can vary.

    The add_offset flag will add an const. offset on all
    const values drawn from an equal distribution

    The add_noise flag will add Gaussian noise to the 
    input current.

    The add_jitter flag will add temporal jitter to
    the step input profiles.
    """

    classes = neuron_parameters.keys()

    # run the enocding and create training data
    encoded_data_original = []

    # create training dataset by iterating over neuron params and input currents
    for _, class_name in enumerate(classes):
        # mkae sure we have at least one spike in the trial
        go_next = False
        while not go_next:
            # iterate over changes
            sim_time = int(runtime[class_name] *
                        (1/neuron_parameters[class_name]["dt"])*1E-3)
            try:
                time_points_local = [int(
                    i*(1/neuron_parameters[class_name]["dt"])*1E-3) for i in time_points[class_name]]
            except:
                KeyError
            # variable input currents over time
            if len(input_currents[class_name]) > 1:
                if time_points[class_name] is None:
                    print('Missing time points.')
                # current with step function
                input_current = np.zeros((sim_time, 1))
                for counter, actual_current in enumerate(input_currents[class_name]):
                    if add_offset:
                        # np.random.random_sample(): Return random floats in the half-open interval [0.0, 1.0).
                        input_current_local = actual_current + \
                            (np.random.random_sample() - 0.5)*offset
                        # if current_tmp < 0.0:
                        #     actual_current = 0.0
                        # else:
                        #     actual_current = current_tmp
                    if add_jitter:
                        # temporal jitter in ms
                        _jitter = np.arange(-int((jitter/2)), int((jitter/2)))
                        _jitter = np.random.choice(_jitter)
                    # new current from t on
                    if add_jitter:
                        t = time_points_local[counter]+_jitter
                        if t < 0:
                            t = 0
                    else:
                        t = time_points_local[counter]
                    input_current[t:] = actual_current
            else:
                # const current
                if add_offset:
                    input_current_local = input_currents[class_name][0] + \
                        (np.random.random_sample() - 0.5)*offset
                    # if current_tmp < 0.0:
                    #     input_current_local = 0.0
                    # else:
                    #     input_current_local = current_tmp
                else:
                    input_current_local = input_currents[class_name]
                input_current = np.ones((sim_time, 1)) * input_current_local

            if add_noise:
                # add noise
                _noise = np.random.normal(
                    loc=0.0, scale=noise, size=input_current.size)
                input_current = np.array(
                    [input_current[x] + _noise[x] for x in range(len(input_current))])

            # set up MN neuron
            neurons = MN_neuron(
                1, neuron_parameters[class_name], dt=neuron_parameters[class_name]["dt"], train=False)
            if class_name == "Spike latency":
                pass
            # compute neuron output
            input = torch.as_tensor(input_current)
            output_v = []
            output_spk = []
            output_thr = []
            for t in range(input.shape[0]):
                out = neurons(input[t])
                # [0] is needed for single neuron
                output_spk.append(out[0].cpu().numpy())
                output_v.append(neurons.state.V[0].cpu().numpy())
                output_thr.append(neurons.state.Thr[0].cpu().numpy())
            if np.sum(output_spk) > 0:
                encoded_data_original.append(
                    [output_spk, output_v, output_thr, input_current])
                go_next = True
            else:
                print(f"No spike at {class_name}. Repeating.")
                go_next = False

    filename = './data/original_mn_output/data_encoding_original'
    if add_noise:
        filename = filename + f'_{noise}_noise'
    if add_jitter:
        filename = filename + f'_{jitter}_jitter'
    if add_offset:
        filename = filename + f'_{offset}_offset'

    encoded_data_original = np.array(encoded_data_original, dtype=object)

    # dump neuron output to file
    with open(f"{filename}.pkl", 'wb') as handle:
        pkl.dump(encoded_data_original, handle,
                 protocol=pkl.HIGHEST_PROTOCOL)


def fix_time_only(max_trials=100):
    """
    Creates data for behavior classes regarding neuron 
    input current (profiles) with fix time duration 
    and neuron parameters for each class. Repetitions
    are created by simply copying the current trace
    n times.
    """

    # import neuron params
    from tactile_encoding.parameters.ideal_params import (input_currents,
                                                          neuron_parameters,
                                                          runtime, time_points)
    max_time = max(runtime.values())  # get max run time (1000ms)
    classes = neuron_parameters.keys()

    # run the enocding and create training data
    encoded_data = []
    encoded_label = []

    # create training dataset by iterating over neuron params and input currents
    for _, class_name in enumerate(classes):
        # print(f'Working on {class_name}')
        # do some sanity checks
        if runtime[class_name] is None:
            print('No runtime given.')
        if input_currents[class_name] is None:
            print('No input current given.')
        # iterate over changes
        sim_time = runtime[class_name]
        # variable input currents over time
        if len(input_currents[class_name]) > 1:
            if time_points[class_name] is None:
                print('Missing time points.')
            input_current = np.zeros((sim_time, 1))
            for counter, actual_current in enumerate(input_currents[class_name]):
                input_current[time_points[class_name]
                              [counter]:] = actual_current
        else:
            # const current
            input_current = np.ones((sim_time, 1)) * \
                input_currents[class_name]

        # stack input current trace if input length < 1000ms
        # round down, with first list initialized
        factor = round((max_time/sim_time)+0.5)
        if factor > 1:
            input_current_list = input_current
            for _ in range(factor):
                input_current_list = np.append(
                    input_current_list, input_current, axis=0)
            input_current = input_current_list

        # set fix input length
        if len(input_current) < max_time:
            print("ERROR-> Trial too short! <-ERROR")
        elif len(input_current) > max_time:
            input_current = input_current[:max_time]

        input = torch.as_tensor(input_current)

        # set up MN neuron
        neurons = MN_neuron(
            1, neuron_parameters[class_name], dt=1E-3, train=False)

        # compute new neuron output
        output_v = []
        output_spk = []
        output_thr = []
        for t in range(input.shape[0]):
            out = neurons(input[t])
            # [0] is needed for single neuron
            output_spk.append(out[0].cpu().numpy())
            output_v.append(neurons.state.V[0].cpu().numpy())
            output_thr.append(neurons.state.Thr[0].cpu().numpy())

        # create max_trials trials per class
        for _ in range(max_trials):
            # store neuron output
            encoded_data.append(
                [output_spk, output_v, output_thr, input_current])
            encoded_label.append(class_name)

    encoded_data = np.array(encoded_data)
    encoded_label = np.array(encoded_label)

    # dump neuron output to file
    with open('./data/original_mn_output/data_encoding_fix_len.pkl', 'wb') as handle:
        pkl.dump(encoded_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open('./data/original_mn_output/label_encoding_fix_len.pkl', 'wb') as handle:
        pkl.dump(encoded_label, handle, protocol=pkl.HIGHEST_PROTOCOL)


def fix_time(max_trials=2, offset=1E-1, noise=1E-1, jitter=10, add_offset=False, add_noise=False, add_jitter=False):
    """
    Preferable over fix_time_only.
    Creates data for behavior classes regarding neuron 
    input current (profiles) with fix time duration 
    and neuron parameters for each class. Repetitions
    are created by simply cpoying the current trace
    n times.

    The add_noise flag will add Gaussian noise to the 
    input current.

    The add_jitter flag will add temporal jitter to
    the step input profiles.
    """
    # import neuron params
    from tactile_encoding.parameters.ideal_params import (input_currents,
                                                          neuron_parameters,
                                                          runtime, time_points)
    # from tactile_encoding.utils.utils import check_cuda

    # device = check_cuda(share_GPU=False, gpu_sel=0, gpu_mem_frac=None)
    max_time = max(runtime.values())  # get max run time (1000ms)

    classes = neuron_parameters.keys()
    encoded_data = []
    encoded_label = []

    # create training dataset by iterating over neuron params and input currents
    for _, class_name in tqdm(enumerate(classes)):

        # print(f'Working on {class_name}')
        # do some sanity checks
        if runtime[class_name] is None:
            print('No runtime given.')
        if input_currents[class_name] is None:
            print('No input current given.')


        # create max_trials trials per class
        for _ in range(max_trials):
            go_next = False
            while not go_next:
                # iterate over changes
                sim_time = runtime[class_name]

                # calc if input length < max_time and duplicate if needed
                if sim_time < max_time:
                    factor = round((max_time/sim_time)+0.5)
                else:
                    factor = 1

                # dynamic input current
                if len(input_currents[class_name]) > 1:
                    if time_points[class_name] is None:
                        print('Missing time points.')

                    input_current = np.zeros((sim_time*factor, 1))

                    # create local copies
                    input_currents_copy = input_currents[class_name].copy()
                    time_points_copy = time_points[class_name].copy()

                    for counter in range(factor-1):
                        input_currents_copy.extend(
                            input_currents[class_name].copy())
                        tmp = [time_points[class_name][x] +
                            (counter+1) * sim_time for x in range(len(time_points[class_name]))]
                        time_points_copy.extend(tmp)

                    if add_jitter:
                        # create temp jitter
                        _jitter = [int(np.random.random()*jitter)
                                for _ in range(len(time_points_copy)-1)]
                        # always start at t=0, initial current
                        for x in range(len(time_points_copy)-1):
                            time_points_copy[x +
                                            1] = time_points_copy[x+1]+_jitter[x]
                        # make sure time_points_copy is const increasing
                        dt_time_points = np.diff(time_points_copy)
                        if np.min(dt_time_points[1:]) <= 0.0:
                            for pos, value in enumerate(dt_time_points):
                                # skip first value
                                if pos != 0 and value <= 0.0:
                                    # found dt <= 0
                                    time_points_copy[pos +
                                                    1] = time_points_copy[pos]+1
                                    # update dt_time_points
                                    dt_time_points = np.diff(time_points_copy)
                    # calc input current trace
                    for counter, actual_current in enumerate(input_currents_copy):
                        if add_offset:
                            input_current_local = actual_current + \
                                (np.random.random_sample() - 0.5)*offset
                            # if current_tmp < 0.0:
                            #     actual_current = 0.0
                            # else:
                            #     actual_current = current_tmp
                        input_current[time_points_copy[counter]:] = actual_current
                # const input current
                else:
                    # const current
                    if add_offset:
                        input_current_local = input_currents[class_name][0] + \
                            (np.random.random_sample() - 0.5)*offset
                        # if current_tmp < 0.0:
                        #     input_current_local = 0.0
                        # else:
                        #     input_current_local = current_tmp
                    else:
                        input_current_local = input_currents[class_name]

                    input_current = np.ones((max_time, 1)) * input_current_local

                # set fix input length
                if len(input_current) < max_time:
                    print("ERROR-> Trial too short! <-ERROR")
                elif len(input_current) > max_time:
                    input_current = input_current[:max_time]

                if add_noise:
                    # add noise on input current
                    _noise = np.random.normal(
                        loc=0.0, scale=noise, size=input_current.size)
                    input_current = np.array(
                        [input_current[x] + _noise[x] for x in range(len(input_current))])

                # convert input current to tensor
                input = torch.as_tensor(input_current)  # .to(device)

                # set up MN neuron
                neurons = MN_neuron(
                    1, neuron_parameters[class_name], dt=1E-3, train=False)

                # compute new neuron output
                output_v = []
                output_spk = []
                output_thr = []
                for t in range(input.shape[0]):
                    out = neurons(input[t])
                    # [0] is needed for single neuron
                    output_spk.append(out[0].cpu().numpy())
                    output_v.append(neurons.state.V[0].cpu().numpy())
                    output_thr.append(neurons.state.Thr[0].cpu().numpy())
                if np.sum(output_spk) > 0:
                    # store neuron output
                    encoded_data.append(
                        [output_spk, output_v, output_thr, input_current])
                    encoded_label.append(class_name)
                    go_next = True
                else:
                    # print(f'Missing spikes in {class_name}')
                    go_next = False
    filename_data = './data/original_mn_output/data_encoding_fix_len'
    filename_label = './data/original_mn_output/label_encoding_fix_len'
    if add_noise:
        filename_data = filename_data + f'_{noise}_noise'
        filename_label = filename_label + '_noisy'
    if add_jitter:
        filename_data = filename_data + f'_{jitter}_jitter'
        filename_label = filename_label + '_jitter'
    if add_offset:
        filename_data = filename_data + f'_{offset}_offset'
        filename_label = filename_label + '_offset'

    encoded_data = np.array(encoded_data)
    encoded_label = np.array(encoded_label)

    # dump neuron output to file
    with open(f"{filename_data}.pkl", 'wb') as handle:
        pkl.dump(encoded_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(f"{filename_label}.pkl", 'wb') as handle:
        pkl.dump(encoded_label, handle, protocol=pkl.HIGHEST_PROTOCOL)


def indices_of_sign_change(data):
    """
    Returns indices of sign change in list.
    """
    idc = []
    for idx in range(0, len(data) - 1):
        # checking for successive opposite index
        if data[idx] > 0 and data[idx + 1] < 0 or data[idx] < 0 and data[idx + 1] > 0:
            idc.append(idx)

    return idc


def _two_scales(ax1, time, data1, data2, data3, data4, c1, c2, c3, c4, create_xlabel=False, create_ylabel1=False, create_ylabel2=False):
    """
    Creates subplot with shared x axis and 2 y axis.
    """
    ax2 = ax1.twinx()

    # convert V to mV
    data1 = data1*1E3
    data3 = data3*1E3

    # get spike times
    spk_idc = np.where(data4 == 1)[0]
    spk_time = time[spk_idc]
    v_spk = data1[spk_idc-1]

    # if we have spikes we want to reset the voltage and threshold trace
    if np.sum(spk_idc) > 0:
        spk_idc_list = []
        spk_idc_list.append(0)
        spk_idc_list.extend(spk_idc)
        spk_idc_list.append(len(data1))

        for i in range(len(spk_idc_list)-1):
            # plot threshold trace
            ax1.plot(time[spk_idc_list[i]: spk_idc_list[i+1]], data3[spk_idc_list[i]: spk_idc_list[i+1]], color=c3, linestyle='-.', alpha=0.7)

            # plot voltage trace
            ax1.plot(time[spk_idc_list[i]: spk_idc_list[i+1]], data1[spk_idc_list[i]: spk_idc_list[i+1]], color=c1)
    else:
        # plot threshold trace
        ax1.plot(time, data3, color=c3, linestyle='-.', alpha=0.7)

        # plot voltage trace
        ax1.plot(time, data1, color=c1)

    # TODO to seperate this at step we would need to save step times...
    # input current trace
    ax2.plot(time, data2, color=c2, alpha=0.7, linewidth=1)

    # spike times at peak of voltage trace
    ax1.scatter(spk_time, v_spk, s=15, color=c4)

    ax2.set_ylim([-4, 9])
    # ax1.set_xticks(time, minor=False)  # fontsize=8
    # create labels if needed
    if create_xlabel:
        ax1.set_xlabel('time [s]', fontsize=AXIS_LABEL_FONTSIZE)
    if create_ylabel1:
        ax1.set_ylabel('voltage [mV]', fontsize=AXIS_LABEL_FONTSIZE)
    if create_ylabel2:
        ax2.set_ylabel('current/C [V/s]', fontsize=AXIS_LABEL_FONTSIZE)

    return ax1, ax2


def plot_traces_original(path, data, add_offset=False, add_noise=False, temp_jitter=False):
    """
    Creates the input current and membrane voltage traces for the original data.
    """

    classes_list = {
        'A': "Tonic spiking",
        'B': "Class 1",
        'C': "Spike frequency adaptation",
        'D': "Phasic spiking",
        'E': "Accommodation",
        'F': "Threshold variability",
        'G': "Rebound spike",
        'H': "Class 2",
        'I': "Integrator",
        'J': "Input bistability",
        'K': "Hyperpolarizing spiking",
        'L': "Hyperpolarizing bursting",
        'M': "Tonic bursting",
        'N': "Phasic bursting",
        'O': "Rebound burst",
        'P': "Mixed mode",
        'Q': "Afterpotentials",
        'R': "Basal bistability",
        'S': "Preferred frequency",
        'T': "Spike latency",
    }

    figname = 'Original traces'
    if add_noise:
        figname = figname + ' - noisy'
    if temp_jitter:
        figname = figname + ' - temp jitter'
    if add_offset:
        figname = figname + ' - offset'
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(figname, fontsize=SUPTITLE_FONT_SIZE)
    for num, el in enumerate(list(classes_list.values())):
        ax = plt.subplot(5, 4, num+1)
        ax.set_title("{}: {}".format(value2key(
            el, classes_list)[0], el), fontsize=TITLE_FONTSIZE)

        # raster plot
        spikes = np.reshape(np.array(data[num][0]), (np.array(
            data[num][0]).shape[0]))

        # voltage trace
        voltage = np.reshape(np.array(data[num][1]), (np.array(
            data[num][1]).shape[0]))

        # threshold trace
        threshold = np.reshape(np.array(data[num][2]), (np.array(
            data[num][2]).shape[0]))

        # input current trace
        input_current = np.reshape(np.array(data[num][3]), (np.array(
            data[num][3]).shape[0]))

        # lets create the correct time axis
        time = np.linspace(0.0, len(voltage)*neuron_parameters[el]["dt"], len(voltage))

        # only add labels on most outer subplot
        # create left y label
        if num == 0 or num == 4 or num == 8 or num == 12:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                   data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=False, create_ylabel1=True, create_ylabel2=False)
        # create right y label
        elif num == 3 or num == 7 or num == 11 or num == 15:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                   data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=False, create_ylabel1=False, create_ylabel2=True)
        # create left y label and x label
        elif num == 16:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                   data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=True, create_ylabel1=True, create_ylabel2=False)
        # create x label
        elif num > 16 and num < 19:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                   data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=True, create_ylabel1=False, create_ylabel2=False)
        # create right y label and x label
        elif num == 19:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                   data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=True, create_ylabel1=False, create_ylabel2=True)
        # create no label
        else:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                   data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=False, create_ylabel1=False, create_ylabel2=False)

    filepath = f'{path}/traces_original'
    if add_noise:
        filepath = filepath + '_noisy'
    if temp_jitter:
        filepath = filepath + '_temp_jitter'
    if add_offset:
        filepath = filepath + '_offset'
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f'{filepath}.pdf')
    plt.close(fig)


def plot_isi_original(path, data, add_offset=False, add_noise=False, temp_jitter=False, norm_count=False, norm_time=False):
    """
    Calculates and plots the ISI for the original data.
    """

    classes_list = {
        'A': "Tonic spiking",
        'B': "Class 1",
        'C': "Spike frequency adaptation",
        'D': "Phasic spiking",
        'E': "Accommodation",
        'F': "Threshold variability",
        'G': "Rebound spike",
        'H': "Class 2",
        'I': "Integrator",
        'J': "Input bistability",
        'K': "Hyperpolarizing spiking",
        'L': "Hyperpolarizing bursting",
        'M': "Tonic bursting",
        'N': "Phasic bursting",
        'O': "Rebound burst",
        'P': "Mixed mode",
        'Q': "Afterpotentials",
        'R': "Basal bistability",
        'S': "Preferred frequency",
        'T': "Spike latency",
    }

    figname = 'ISI original'
    if add_noise:
        figname = figname + ' - noisy'
    if temp_jitter:
        figname = figname + ' - temp jitter'
    if add_offset:
        figname = figname + ' - offset'
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(figname, fontsize=SUPTITLE_FONT_SIZE)
    for num, el in enumerate(list(classes_list.values())):
        ax = plt.subplot(5, 4, num+1)
        ax.set_title("{}: {}".format(value2key(
            el, classes_list)[0], el), fontsize=TITLE_FONTSIZE)
        # TODO inlcude grid to all plots
        # plt.grid()

        # cal ISI
        spikes = np.reshape(np.array(data[num][0]), (np.array(
            data[num][0]).shape[0]))
        isi_original = np.diff(np.where(spikes == 1)[0])

        # calc hist if ISI found
        if len(isi_original) > 0:
            tmp_original = np.unique(isi_original, return_counts=True)
            isi_original = tmp_original[0]
            if norm_time:
                isi_original = isi_original/max(isi_original)
            isi_original_count = tmp_original[1]
            if norm_count:
                isi_original_count = isi_original_count/max(isi_original_count)
            if norm_time:
                ax.bar(isi_original, isi_original_count, width=0.01)
                ax.set_xlim((0, 1.1))
            else:
                ax.bar(isi_original, isi_original_count)
        else:
            ax.text(0.3, 0.5, f'nbr. spikes = {len(np.where(spikes == 1))}')
        plt.tick_params(axis='x', labelsize=TICK_FONTSIZE)
        plt.tick_params(axis='y', labelsize=TICK_FONTSIZE)

        # only add labels on most outer subplot
        if num == 0 or num == 4 or num == 8 or num == 12:
            if norm_count:
                ax.set_ylabel('normalized count', fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_ylabel('count', fontsize=AXIS_LABEL_FONTSIZE)
        elif num == 16:
            if norm_time:
                ax.set_xlabel('normalized ISI', fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_xlabel('ISI [ms]', fontsize=AXIS_LABEL_FONTSIZE)
            if norm_count:
                ax.set_ylabel('normalized count', fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_ylabel('count', fontsize=AXIS_LABEL_FONTSIZE)
        elif num > 16:
            if norm_time:
                ax.set_xlabel('normalized ISI', fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_xlabel('ISI [ms]', fontsize=AXIS_LABEL_FONTSIZE)

    filepath = f'{path}/isi_original'
    if add_noise:
        filepath = filepath + '_noisy'
    if temp_jitter:
        filepath = filepath + '_temp_jitter'
    if add_offset:
        filepath = filepath + '_offset'

    fig.tight_layout()
    fig.savefig(f'{filepath}.pdf')
    plt.close(fig)


def plot_traces_fix_len(path, data, max_trials, add_offset=False, add_noise=False, temp_jitter=False):
    """
    Creates the input current and membrane voltage traces for the data with fix length.
    """

    classes_list = {
        'A': "Tonic spiking",
        'B': "Class 1",
        'C': "Spike frequency adaptation",
        'D': "Phasic spiking",
        'E': "Accommodation",
        'F': "Threshold variability",
        'G': "Rebound spike",
        'H': "Class 2",
        'I': "Integrator",
        'J': "Input bistability",
        'K': "Hyperpolarizing spiking",
        'L': "Hyperpolarizing bursting",
        'M': "Tonic bursting",
        'N': "Phasic bursting",
        'O': "Rebound burst",
        'P': "Mixed mode",
        'Q': "Afterpotentials",
        'R': "Basal bistability",
        'S': "Preferred frequency",
        'T': "Spike latency",
    }

    figname = 'Fix length traces'
    if add_noise:
        figname = figname + ' - noisy'
    if temp_jitter:
        figname = figname + ' - temp jitter'
    if add_offset:
        figname = figname + ' - offset'
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(figname, fontsize=SUPTITLE_FONT_SIZE)
    for num, el in enumerate(list(classes_list.values())):
        # lets visualize all trials
        # select a sample trial out of max_trials
        pos = range(num*max_trials, num*max_trials+max_trials)
        pos = np.random.choice(pos)

        ax = plt.subplot(5, 4, num+1)
        ax.set_title("{}: {}".format(value2key(
            el, classes_list)[0], el), fontsize=TITLE_FONTSIZE)

        # get spike times
        spikes = np.reshape(np.array(data[pos][0]), (np.array(
            data[pos][0]).shape[0]))

        # voltage trace
        voltage = np.reshape(np.array(data[pos][1]), (np.array(
            data[pos][1]).shape[0]))

        # threshold trace
        threshold = np.reshape(np.array(data[pos][2]), (np.array(
            data[pos][2]).shape[0]))

        # input current trace
        input_current = np.reshape(np.array(data[pos][3]), (np.array(
            data[pos][3]).shape[0]))

        # lets create the correct time axis
        time = np.linspace(0.0, len(voltage)*neuron_parameters[el]["dt"], len(voltage))

        # only add labels on most outer subplot
        # create left y label
        if num == 0 or num == 4 or num == 8 or num == 12:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                    data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=False, create_ylabel1=True, create_ylabel2=False)
        # create right y label
        elif num == 3 or num == 7 or num == 11 or num == 15:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                    data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=False, create_ylabel1=False, create_ylabel2=True)
        # create left y label and x label
        elif num == 16:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                    data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=True, create_ylabel1=True, create_ylabel2=False)
        # create x label
        elif num > 16 and num < 19:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                    data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=True, create_ylabel1=False, create_ylabel2=False)
        # create right y label and x label
        elif num == 19:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                    data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=True, create_ylabel1=False, create_ylabel2=True)
        # create no label
        else:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                    data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=False, create_ylabel1=False, create_ylabel2=False)

    filepath = f'{path}/traces_fix_len'
    if add_noise:
        filepath = filepath + '_noisy'
    if temp_jitter:
        filepath = filepath + '_temp_jitter'
    if add_offset:
        filepath = filepath + '_offset'

    fig.tight_layout()
    fig.savefig(f'{filepath}.pdf')
    plt.close(fig)


def plot_single_isi_fix_len(path, data, max_trials, add_offset=False, add_noise=False, temp_jitter=False, norm_time=False, norm_count=False):
    """
    Calculates and plots the ISI for a single random trial of the fix length data.
    """

    classes_list = {
        'A': "Tonic spiking",
        'B': "Class 1",
        'C': "Spike frequency adaptation",
        'D': "Phasic spiking",
        'E': "Accommodation",
        'F': "Threshold variability",
        'G': "Rebound spike",
        'H': "Class 2",
        'I': "Integrator",
        'J': "Input bistability",
        'K': "Hyperpolarizing spiking",
        'L': "Hyperpolarizing bursting",
        'M': "Tonic bursting",
        'N': "Phasic bursting",
        'O': "Rebound burst",
        'P': "Mixed mode",
        'Q': "Afterpotentials",
        'R': "Basal bistability",
        'S': "Preferred frequency",
        'T': "Spike latency",
    }

    figname = 'ISI fix length single'
    if add_noise:
        figname = figname + ' - noisy'
    if temp_jitter:
        figname = figname + ' - temp jitter'
    if add_offset:
        figname = figname + ' - offset'
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(figname, fontsize=SUPTITLE_FONT_SIZE)
    for num, el in enumerate(list(classes_list.values())):
        # select a sample trial out of max_trials
        pos = range(num*max_trials, num*max_trials+max_trials)
        pos = np.random.choice(pos)

        ax = plt.subplot(5, 4, num+1)
        ax.set_title("{}: {}".format(value2key(
            el, classes_list)[0], el), fontsize=TITLE_FONTSIZE)
        # TODO inlcude grid to all plots
        # plt.grid()

        # calc ISI
        spikes = np.reshape(np.array(data[pos][0]), (np.array(
            data[pos][0]).shape[0]))
        isi_fix_len = np.diff(np.where(spikes == 1)[0])

        # calc hist if ISI found
        if len(isi_fix_len) > 0:
            tmp_fix_len = np.unique(isi_fix_len, return_counts=True)
            isi_fix_len = tmp_fix_len[0]
            if norm_time:
                isi_fix_len = isi_fix_len/max(isi_fix_len)
            isi_fix_len_count = tmp_fix_len[1]
            if norm_count:
                isi_fix_len_count = isi_fix_len_count/max(isi_fix_len_count)
            if norm_time:
                ax.bar(isi_fix_len, isi_fix_len_count, width=0.01)
                ax.set_xlim((0, 1.1))
            else:
                ax.bar(isi_fix_len, isi_fix_len_count)
        else:
            ax.text(0.3, 0.5, f'nbr. spikes = {len(np.where(spikes == 1))}')
        plt.tick_params(axis='x', labelsize=6)
        plt.tick_params(axis='y', labelsize=6)

        # only add labels on most outer subplot
        if num == 0 or num == 4 or num == 8 or num == 12:
            if norm_count:
                ax.set_ylabel('normalized count', fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_ylabel('count', fontsize=AXIS_LABEL_FONTSIZE)
        elif num == 16:
            if norm_time:
                ax.set_xlabel('normalized ISI', fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_xlabel('ISI [ms]', fontsize=AXIS_LABEL_FONTSIZE)
            if norm_count:
                ax.set_ylabel('normalized count', fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_ylabel('count', fontsize=AXIS_LABEL_FONTSIZE)
        elif num > 16:
            if norm_time:
                ax.set_xlabel('normalized ISI', fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_xlabel('ISI [ms]', fontsize=AXIS_LABEL_FONTSIZE)

    filepath = f'{path}/isi_single_trial_fix_len'
    if norm_count:
        filepath = filepath + '_norm_count'
    if norm_count:
        filepath = filepath + '_norm_time'
    if add_noise:
        filepath = filepath + '_noisy'
    if temp_jitter:
        filepath = filepath + '_temp_jitter'
    if add_offset:
        filepath = filepath + '_offset'

    fig.tight_layout()
    fig.savefig(f'{filepath}.pdf')
    plt.close(fig)


def plot_isi_fix_len(path, data, max_trials, add_offset=False, add_noise=False, temp_jitter=False, norm_count=False, norm_time=False):
    """
    Calculates and plots the ISI for all repetitions of fix length data.
    """

    classes_list = {
        'A': "Tonic spiking",
        'B': "Class 1",
        'C': "Spike frequency adaptation",
        'D': "Phasic spiking",
        'E': "Accommodation",
        'F': "Threshold variability",
        'G': "Rebound spike",
        'H': "Class 2",
        'I': "Integrator",
        'J': "Input bistability",
        'K': "Hyperpolarizing spiking",
        'L': "Hyperpolarizing bursting",
        'M': "Tonic bursting",
        'N': "Phasic bursting",
        'O': "Rebound burst",
        'P': "Mixed mode",
        'Q': "Afterpotentials",
        'R': "Basal bistability",
        'S': "Preferred frequency",
        'T': "Spike latency",
    }

    figname = 'ISI fix length all'
    if add_noise:
        figname = figname + ' - noisy'
    if temp_jitter:
        figname = figname + ' - temp jitter'
    if add_offset:
        figname = figname + ' - offset'
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(figname, fontsize=SUPTITLE_FONT_SIZE)
    for num, el in enumerate(list(classes_list.values())):
        # concatenate all ISIs
        isi_fix_len = []
        for trial in range(max_trials):
            # calc spikes per trial
            spikes = np.reshape(np.array(data[trial + num*max_trials][0]), (np.array(
                data[trial + num*max_trials][0]).shape[0]))
            # calc ISI
            isi_fix_len.extend(np.diff(np.where(spikes == 1)[0]))

        ax = plt.subplot(5, 4, num+1)
        ax.set_title("{}: {}".format(value2key(
            el, classes_list)[0], el), fontsize=TITLE_FONTSIZE)

        # TODO inlcude grid to all plots
        # plt.grid()

        if len(isi_fix_len) > 0:
            tmp_fix_len = np.unique(isi_fix_len, return_counts=True)
            isi_fix_len = tmp_fix_len[0]
            if norm_time:
                isi_fix_len = isi_fix_len/max(isi_fix_len)
            isi_fix_len_count = tmp_fix_len[1]
            if norm_count:
                isi_fix_len_count = isi_fix_len_count/max(isi_fix_len_count)
            if norm_time:
                ax.bar(isi_fix_len, isi_fix_len_count, width=0.01)
                ax.set_xlim((0, 1.1))
            else:
                ax.bar(isi_fix_len, isi_fix_len_count)
        else:
            ax.text(0.3, 0.5, f'nbr. spikes = {len(np.where(spikes == 1))}')

        plt.tick_params(axis='x', labelsize=6)
        plt.tick_params(axis='y', labelsize=6)

        # only add labels on most outer subplot
        if num == 0 or num == 4 or num == 8 or num == 12:
            if norm_count:
                ax.set_ylabel('count')
            else:
                ax.set_ylabel('count')
        elif num == 16:
            if norm_time:
                ax.set_xlabel('ISI')
            else:
                ax.set_xlabel('ISI [ms]')
            if norm_count:
                ax.set_ylabel('count')
            else:
                ax.set_ylabel('count')
        elif num > 16:
            if norm_time:
                ax.set_xlabel('ISI')
            else:
                ax.set_xlabel('ISI [ms]')

    filepath = f'{path}/isi_fix_len'
    if add_noise:
        filepath = filepath + '_noisy'
    if temp_jitter:
        filepath = filepath + '_temp_jitter'
    if add_offset:
        filepath = filepath + '_offset'

    fig.tight_layout()
    fig.savefig(f'{filepath}.pdf')
    plt.close(fig)


def plot_traces_fix_len_param_sweep(path, data, max_trials, offset=0.1, noise=0.1, jitter=10, add_offset=False, add_noise=False, temp_jitter=False):
    """
    Creates the input current and membrane voltage traces for the data with fix length.
    """

    classes_list = {
        'A': "Tonic spiking",
        'B': "Class 1",
        'C': "Spike frequency adaptation",
        'D': "Phasic spiking",
        'E': "Accommodation",
        'F': "Threshold variability",
        'G': "Rebound spike",
        'H': "Class 2",
        'I': "Integrator",
        'J': "Input bistability",
        'K': "Hyperpolarizing spiking",
        'L': "Hyperpolarizing bursting",
        'M': "Tonic bursting",
        'N': "Phasic bursting",
        'O': "Rebound burst",
        'P': "Mixed mode",
        'Q': "Afterpotentials",
        'R': "Basal bistability",
        'S': "Preferred frequency",
        'T': "Spike latency",
    }

    figname = f'{noise} noise, {jitter} jitter, {offset} offset'
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(figname, fontsize=SUPTITLE_FONT_SIZE)
    for num, el in enumerate(list(classes_list.values())):
        # select a sample trial out of max_trials
        pos = range(num*max_trials, num*max_trials+max_trials)
        pos = np.random.choice(pos)

        ax = plt.subplot(5, 4, num+1)
        ax.set_title("{}: {}".format(value2key(
            el, classes_list)[0], el), fontsize=TITLE_FONTSIZE)

        # get spike times
        spikes = np.reshape(np.array(data[pos][0]), (np.array(
            data[pos][0]).shape[0]))

        # voltage trace
        voltage = np.reshape(np.array(data[pos][1]), (np.array(
            data[pos][1]).shape[0]))

        # threshold trace
        threshold = np.reshape(np.array(data[pos][2]), (np.array(
            data[pos][2]).shape[0]))

        # input current trace
        input_current = np.reshape(np.array(data[pos][3]), (np.array(
            data[pos][3]).shape[0]))

        # lets create the correct time axis
        time = np.linspace(0.0, len(voltage)*neuron_parameters[el]["dt"], len(voltage))

        # only add labels on most outer subplot
        # create left y label
        if num == 0 or num == 4 or num == 8 or num == 12:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                   data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=False, create_ylabel1=True, create_ylabel2=False)
        # create right y label
        elif num == 3 or num == 7 or num == 11 or num == 15:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                   data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=False, create_ylabel1=False, create_ylabel2=True)
        # create left y label and x label
        elif num == 16:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                   data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=True, create_ylabel1=True, create_ylabel2=False)
        # create x label
        elif num > 16 and num < 19:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                   data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=True, create_ylabel1=False, create_ylabel2=False)
        # create right y label and x label
        elif num == 19:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                   data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=True, create_ylabel1=False, create_ylabel2=True)
        # create no label
        else:
            ax1, ax2 = _two_scales(ax1=ax, time=time, data1=voltage, data2=input_current, data3=threshold,
                                   data4=spikes, c1='b', c2='orange', c3='k', c4='r', create_xlabel=False, create_ylabel1=False, create_ylabel2=False)

    filepath = f'{path}/traces_fix_len'
    if add_noise:
        filepath = filepath + f'_{noise}_noise'
    if temp_jitter:
        filepath = filepath + f'_{jitter}_jitter'
    if add_offset:
        filepath = filepath + f'_{offset}_offset'

    fig.tight_layout()
    fig.savefig(f'{filepath}.pdf')
    plt.close(fig)


def plot_single_isi_fix_len_param_sweep(path, data, max_trials, offset=0.1, noise=0.1, jitter=10, add_offset=False, add_noise=False, temp_jitter=False, norm_time=False, norm_count=False):
    """
    Calculates and plots the ISI for a single random trial of the fix length data.
    """

    classes_list = {
        'A': "Tonic spiking",
        'B': "Class 1",
        'C': "Spike frequency adaptation",
        'D': "Phasic spiking",
        'E': "Accommodation",
        'F': "Threshold variability",
        'G': "Rebound spike",
        'H': "Class 2",
        'I': "Integrator",
        'J': "Input bistability",
        'K': "Hyperpolarizing spiking",
        'L': "Hyperpolarizing bursting",
        'M': "Tonic bursting",
        'N': "Phasic bursting",
        'O': "Rebound burst",
        'P': "Mixed mode",
        'Q': "Afterpotentials",
        'R': "Basal bistability",
        'S': "Preferred frequency",
        'T': "Spike latency",
    }

    figname = f'{noise} noise, {jitter} jitter, {offset} offset'
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(figname, fontsize=SUPTITLE_FONT_SIZE)
    for num, el in enumerate(list(classes_list.values())):
        # select a sample trial out of max_trials
        pos = range(num*max_trials, num*max_trials+max_trials)
        pos = np.random.choice(pos)

        ax = plt.subplot(5, 4, num+1)
        ax.set_title("{}: {}".format(value2key(
            el, classes_list)[0], el), fontsize=TITLE_FONTSIZE)
        # TODO inlcude grid to all plots
        # plt.grid()

        # calc ISI
        spikes = np.reshape(np.array(data[pos][0]), (np.array(
            data[pos][0]).shape[0]))
        isi_fix_len = np.diff(np.where(spikes == 1)[0])

        # calc hist if ISI found
        if len(isi_fix_len) > 0:
            tmp_fix_len = np.unique(isi_fix_len, return_counts=True)
            isi_fix_len = tmp_fix_len[0]
            if norm_time:
                isi_fix_len = isi_fix_len/max(isi_fix_len)
            isi_fix_len_count = tmp_fix_len[1]
            if norm_count:
                isi_fix_len_count = isi_fix_len_count/max(isi_fix_len_count)
            if norm_time:
                ax.bar(isi_fix_len, isi_fix_len_count, width=0.01)
                ax.set_xlim((0, 1.1))
            else:
                ax.bar(isi_fix_len, isi_fix_len_count)
        else:
            ax.text(0.3, 0.5, f'nbr. spikes = {len(np.where(spikes == 1))}')
        plt.tick_params(axis='x', labelsize=6)
        plt.tick_params(axis='y', labelsize=6)

        # only add labels on most outer subplot
        if num == 0 or num == 4 or num == 8 or num == 12:
            if norm_count:
                ax.set_ylabel('count')
            else:
                ax.set_ylabel('count')
        elif num == 16:
            if norm_time:
                ax.set_xlabel('ISI')
            else:
                ax.set_xlabel('ISI [ms]')
            if norm_count:
                ax.set_ylabel('count')
            else:
                ax.set_ylabel('count')
        elif num > 16:
            if norm_time:
                ax.set_xlabel('ISI')
            else:
                ax.set_xlabel('ISI [ms]')

    filepath = f'{path}/isi_single_trial_fix_len'

    if add_noise:
        filepath = filepath + f'_{noise}_noise'
    if temp_jitter:
        filepath = filepath + f'_{jitter}_jitter'
    if add_offset:
        filepath = filepath + f'_{offset}_offset'

    fig.tight_layout()
    fig.savefig(f'{filepath}.pdf')
    plt.close(fig)


def plot_isi_fix_len_param_sweep(path, data, max_trials, offset=0.1, noise=0.1, jitter=10, add_offset=False, add_noise=False, temp_jitter=False, norm_count=False, norm_time=False):
    """
    Calculates and plots the ISI for all repetitions of fix length data.
    """

    classes_list = {
        'A': "Tonic spiking",
        'B': "Class 1",
        'C': "Spike frequency adaptation",
        'D': "Phasic spiking",
        'E': "Accommodation",
        'F': "Threshold variability",
        'G': "Rebound spike",
        'H': "Class 2",
        'I': "Integrator",
        'J': "Input bistability",
        'K': "Hyperpolarizing spiking",
        'L': "Hyperpolarizing bursting",
        'M': "Tonic bursting",
        'N': "Phasic bursting",
        'O': "Rebound burst",
        'P': "Mixed mode",
        'Q': "Afterpotentials",
        'R': "Basal bistability",
        'S': "Preferred frequency",
        'T': "Spike latency",
    }

    figname = f'{noise} noise, {jitter} jitter, {offset} offset'
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(figname, fontsize=SUPTITLE_FONT_SIZE)
    for num, el in enumerate(list(classes_list.values())):
        # concatenate all ISIs
        isi_fix_len = []
        for trial in range(max_trials):
            # calc spikes per trial
            spikes = np.reshape(np.array(data[trial + num*max_trials][0]), (np.array(
                data[trial + num*max_trials][0]).shape[0]))
            # calc ISI
            isi_fix_len.extend(np.diff(np.where(spikes == 1)[0]))

        ax = plt.subplot(5, 4, num+1)
        ax.set_title("{}: {}".format(value2key(
            el, classes_list)[0], el), fontsize=TITLE_FONTSIZE)

        # TODO inlcude grid to all plots
        # plt.grid()

        if len(isi_fix_len) > 0:
            tmp_fix_len = np.unique(isi_fix_len, return_counts=True)
            isi_fix_len = tmp_fix_len[0]
            if norm_time:
                isi_fix_len = isi_fix_len/max(isi_fix_len)
            isi_fix_len_count = tmp_fix_len[1]
            if norm_count:
                isi_fix_len_count = isi_fix_len_count/max(isi_fix_len_count)
            if norm_time:
                ax.bar(isi_fix_len, isi_fix_len_count, width=0.01)
                ax.set_xlim((0, 1.1))
            else:
                ax.bar(isi_fix_len, isi_fix_len_count)
        else:
            ax.text(0.3, 0.5, f'nbr. spikes = {len(np.where(spikes == 1))}')

        plt.tick_params(axis='x', labelsize=6)
        plt.tick_params(axis='y', labelsize=6)

        # only add labels on most outer subplot
        if num == 0 or num == 4 or num == 8 or num == 12:
            if norm_count:
                ax.set_ylabel('count')
            else:
                ax.set_ylabel('count')
        elif num == 16:
            if norm_time:
                ax.set_xlabel('ISI')
            else:
                ax.set_xlabel('ISI [ms]')
            if norm_count:
                ax.set_ylabel('count')
            else:
                ax.set_ylabel('count')
        elif num > 16:
            if norm_time:
                ax.set_xlabel('ISI')
            else:
                ax.set_xlabel('ISI [ms]')

    filepath = f'{path}/isi_fix_len'
    if add_noise:
        filepath = filepath + f'_{noise}_noise'
    if temp_jitter:
        filepath = filepath + f'_{jitter}_jitter'
    if add_offset:
        filepath = filepath + f'_{offset}_offset'

    fig.tight_layout()
    fig.savefig(f'{filepath}.pdf')
    plt.close(fig)


def return_isi_fix_len(data, max_trials, norm_count=False, norm_time=False):
    """
    Calculates and returns the ISI for all repetitions of fix length data.
    """
    classes_list = {
        'A': "Tonic spiking",
        'B': "Class 1",
        'C': "Spike frequency adaptation",
        'D': "Phasic spiking",
        'E': "Accommodation",
        'F': "Threshold variability",
        'G': "Rebound spike",
        'H': "Class 2",
        'I': "Integrator",
        'J': "Input bistability",
        'K': "Hyperpolarizing spiking",
        'L': "Hyperpolarizing bursting",
        'M': "Tonic bursting",
        'N': "Phasic bursting",
        'O': "Rebound burst",
        'P': "Mixed mode",
        'Q': "Afterpotentials",
        'R': "Basal bistability",
        'S': "Preferred frequency",
        'T': "Spike latency",
    }

    isi_list = []
    for num, el in enumerate(list(classes_list.values())):
        # print(el)
        # concatenate all ISIs
        isi_fix_len = []
        for trial in range(max_trials):
            # calc spikes per trial
            spikes = np.reshape(np.array(data[trial + num*max_trials][0]), (np.array(
                data[trial + num*max_trials][0]).shape[0]))
            # calc ISI
            isi_fix_len.extend(np.diff(np.where(spikes == 1)[0]))

        if len(isi_fix_len) > 0:
            tmp_fix_len = np.unique(isi_fix_len, return_counts=True)
            isi_fix_len = tmp_fix_len[0]
            if norm_time:
                isi_fix_len = isi_fix_len/max(isi_fix_len)
            isi_fix_len_count = tmp_fix_len[1]
            if norm_count:
                isi_fix_len_count = isi_fix_len_count/max(isi_fix_len_count)
            # create 2d array
            isi = np.vstack([isi_fix_len, isi_fix_len_count])
        else:
            isi = [[], []]
        isi_list.append(isi)

    return isi_list
