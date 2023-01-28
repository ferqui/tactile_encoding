"""
Creates data according to the paper "A Generalized Linear 
Integrate-and-Fire Neural Model Produces Diverse Spiking Behaviors"
by Stefan Mihalas and Ernst Niebur.

Istituto Italiano di Tecnologia - IIT
Event-driven perception in robotics - EDPR
Genova, Italy

Simon F. Mueller-Cleve
"""


import numpy as np
import pickle as pkl

import torch

from tactile_encoding.utils.models import MN_neuron


def original(add_noise=False, temp_jitter=False):
    """
    Recreates the original data for behavior classes 
    regarding neuron input current (profiles) over time 
    and neuron parameters. For different classes the 
    length can vary.
    
    The add_noise flag will add Gaussian noise to the 
    input current.

    The temp_jitter flag will add temporal jitter to
    the step input profiles.
    """
    # import neuron params
    from tactile_encoding.parameters.ideal_params import neuron_parameters, input_currents, time_points, runtime

    classes = neuron_parameters.keys()

    # run the enocding and create training data
    encoded_data_original = []

    # create training dataset by iterating over neuron params and input currents
    for _, class_name in enumerate(classes):
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
                if temp_jitter:
                    jitter = range(-5, 6) # temporal jitter in ms 
                    jitter = np.random.choice(jitter)
                # new current from t on
                if temp_jitter:
                    t = time_points[class_name][counter]+jitter
                    if t < 0:
                        t = 0
                else:
                    t = time_points[class_name][counter]
                input_current[t:] = actual_current
        else:
            # const current
            input_current = np.ones((sim_time, 1)) * \
                input_currents[class_name]

        # set up MN neuron
        neurons = MN_neuron(
            1, neuron_parameters[class_name], dt=1E-3, train=False)

        if add_noise:
            # add noise
            noise = np.random.normal(
                loc=0.0, scale=0.1, size=input_current.size)
            input_current = np.array(
                [input_current[x] + noise[x] for x in range(len(input_current))])

        # compute neuron output
        input = torch.as_tensor(input_current)
        output_v = []
        output_s = []
        for t in range(input.shape[0]):
            out = neurons(input[t])
            # [0] is needed for single neuron
            output_s.append(out[0].cpu().numpy())
            output_v.append(neurons.state.V[0].cpu().numpy())
        encoded_data_original.append([output_s, output_v, input_current])

    filename = './data/data_encoding_original'
    if add_noise:
        filename = filename + '_noisy'
    if temp_jitter:
        filename = filename + '_temp_jitter'


    encoded_data_original = np.array(encoded_data_original)

    # dump neuron output to file
    with open(f"{filename}.pkl", 'wb') as handle:
        pkl.dump(encoded_data_original, handle,
                    protocol=pkl.HIGHEST_PROTOCOL)


def fix_time_only(max_trials=100, max_time=1000):
    """
    Creates data for behavior classes regarding neuron 
    input current (profiles) with fix time duration 
    and neuron parameters for each class. Repetitions
    are created by simply copying the current trace
    n times.
    """
    # import neuron params
    from tactile_encoding.parameters.ideal_params import neuron_parameters, input_currents, time_points, runtime

    classes = neuron_parameters.keys()

    # run the enocding and create training data
    encoded_data = []
    encoded_label = []

    # create training dataset by iterating over neuron params and input currents
    for _, class_name in enumerate(classes):
        print(f'Working on {class_name}')
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
                input_current[time_points[class_name][counter]:] = actual_current
        else:
            # const current
            input_current = np.ones((sim_time, 1)) * \
                input_currents[class_name]

        # set up MN neuron
        neurons = MN_neuron(
            1, neuron_parameters[class_name], dt=1E-3, train=False)

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

        # compute new neuron output
        output_v = []
        output_s = []
        for t in range(input.shape[0]):
            out = neurons(input[t])
            # [0] is needed for single neuron
            output_s.append(out[0].cpu().numpy())
            output_v.append(neurons.state.V[0].cpu().numpy())

        # create max_trials trials per class
        for _ in range(max_trials):
            # store neuron output
            encoded_data.append([output_s, output_v, input_current])
            encoded_label.append(class_name)

    encoded_data = np.array(encoded_data)
    encoded_label = np.array(encoded_label)

    # dump neuron output to file
    with open('./data/data_encoding.pkl', 'wb') as handle:
        pkl.dump(encoded_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open('./data/label_encoding.pkl', 'wb') as handle:
        pkl.dump(encoded_label, handle, protocol=pkl.HIGHEST_PROTOCOL)


def fix_time(max_trials=100, max_time=1000, add_noise = False, temp_jitter=False):
    """
    Preferable over fix_time_only.
    Creates data for behavior classes regarding neuron 
    input current (profiles) with fix time duration 
    and neuron parameters for each class. Repetitions
    are created by simply cpoying the current trace
    n times.
    
    The add_noise flag will add Gaussian noise to the 
    input current.

    The temp_jitter flag will add temporal jitter to
    the step input profiles.
    """
    # import neuron params
    from tactile_encoding.parameters.ideal_params import neuron_parameters, input_currents, time_points, runtime

    # some intits
    if add_noise:
        noise_level = 0.1 # 0.1 mV
    if temp_jitter:
        jitter_level = 5 # 5 ms

    classes = neuron_parameters.keys()
    encoded_data = []
    encoded_label = []

    # create training dataset by iterating over neuron params and input currents
    for _, class_name in enumerate(classes):
        print(f'Working on {class_name}')
        # do some sanity checks
        if runtime[class_name] is None:
            print('No runtime given.')
        if input_currents[class_name] is None:
            print('No input current given.')

        # create max_trials trials per class
        for _ in range(max_trials):
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
                    input_currents_copy.extend(input_currents[class_name].copy())
                    tmp = [time_points[class_name][x] + (counter+1) * sim_time for x in range(len(time_points[class_name]))]
                    time_points_copy.extend(tmp)

                if temp_jitter:
                    # create temp jitter
                    jitter = [int(np.random.random()*jitter_level) for _ in range(len(time_points_copy)-1)]
                    # always start at t=0, initial current
                    for x in range(len(time_points_copy)-1):
                        time_points_copy[x+1] =  time_points_copy[x+1]+jitter[x]
                    # make sure time_points_copy is const increasing
                    dt_time_points = np.diff(time_points_copy)
                    if np.min(dt_time_points[1:]) <= 0.0:
                        for pos, value in enumerate(dt_time_points):
                            # skip first value
                            if pos != 0 and value <= 0.0:
                                # found dt <= 0
                                time_points_copy[pos+1] = time_points_copy[pos]+1
                                # update dt_time_points
                                dt_time_points = np.diff(time_points_copy)
                # calc input current trace
                for counter, actual_current in enumerate(input_currents_copy):
                    input_current[time_points_copy[counter]:] = actual_current
            # const input current
            else:
                input_current = np.ones((max_time, 1)) * \
                    input_currents[class_name]
            
            # set fix input length
            if len(input_current) < max_time:
                print("ERROR-> Trial too short! <-ERROR")
            elif len(input_current) > max_time:
                input_current = input_current[:max_time]

            if add_noise:
                # add noise on input current
                noise = np.random.normal(
                    loc=0.0, scale=noise_level, size=input_current.size)
                input_current = np.array(
                    [input_current[x] + noise[x] for x in range(len(input_current))])

            # convert input current to tensor
            input = torch.as_tensor(input_current)

            # set up MN neuron
            neurons = MN_neuron(
                1, neuron_parameters[class_name], dt=1E-3, train=False)

            # compute new neuron output
            output_v = []
            output_s = []
            for t in range(input.shape[0]):
                out = neurons(input[t])
                # [0] is needed for single neuron
                output_s.append(out[0].cpu().numpy())
                output_v.append(neurons.state.V[0].cpu().numpy())

            # store neuron output
            encoded_data.append([output_s, output_v, input_current])
            encoded_label.append(class_name)

    filename_data = './data/data_encoding'
    filename_label = './data/label_encoding'
    if add_noise:
        filename_data = filename_data + '_noisy'
        filename_label = filename_label + '_noisy'
    if temp_jitter:
        filename_data = filename_data + '_temp_jitter'
        filename_label = filename_label + '_temp_jitter'

    encoded_data = np.array(encoded_data)
    encoded_label = np.array(encoded_label)

    # dump neuron output to file
    with open(f"{filename_data}.pkl", 'wb') as handle:
        pkl.dump(encoded_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(f"{filename_label}.pkl", 'wb') as handle:
        pkl.dump(encoded_label, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    print('Creating original data.')
    original()

    print('Creating noisy original data.')
    original(add_noise=True, temp_jitter=False)

    print('Creating original data with temporal jitter.')
    original(add_noise=False, temp_jitter=True)

    print('Creating noisy original data with temporal jitter.')
    original(add_noise=True, temp_jitter=True)

    # fix 1000ms length
    print('Creating 1000ms data.')
    fix_time_only() # much faster, 'cause current profile only copied

    print('Creating noisy 1000ms data.')
    fix_time(add_noise=True, temp_jitter=False)

    print('Creating 1000ms data with temporal jitter.')
    fix_time(add_noise=False, temp_jitter=True)

    print('Creating noisy 1000ms data with temporal jitter.')
    fix_time(add_noise=True, temp_jitter=True)


    print('Finished with data creation.')