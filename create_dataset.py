import numpy as np
import pickle as pkl

import torch

from tactile_encoding.utils.models import MN_neuron

def original(add_noise=False, temp_jitter=False):
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
                if counter == 0:
                    input_current[:time_points[class_name]
                                  [counter]] = actual_current

                elif counter < len(time_points[class_name]):
                    input_current[time_points[class_name][counter-1]:time_points[class_name][counter]] = actual_current

                elif counter < len(time_points[class_name]):
                    input_current[time_points[class_name]
                                  [counter-1]:] = actual_current
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

    filename = '../data/data_encoding_original'
    if add_noise:
        filename = filename + '_noisy'
    if temp_jitter:
        filename = filename + '_temp_jitter'

    encoded_data_original = np.array(encoded_data_original)

    # dump neuron output to file
    with open(filename, 'wb') as handle:
        pkl.dump(encoded_data_original, handle,
                    protocol=pkl.HIGHEST_PROTOCOL)


def fix_time():
    # import neuron params
    from tactile_encoding.parameters.ideal_params import neuron_parameters, input_currents, time_points, runtime

    classes = neuron_parameters.keys()

    # run the enocding and create training data
    max_trials = 100
    max_time = 1000  # ms
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
                if counter == 0:
                    input_current[:time_points[class_name]
                                    [counter]] = actual_current

                elif counter < len(time_points[class_name]):
                    input_current[time_points[class_name][counter-1]
                        :time_points[class_name][counter]] = actual_current

                elif counter < len(time_points[class_name]):
                    input_current[time_points[class_name]
                                    [counter-1]:] = actual_current
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

            # cut down to fix length
            input_current = np.array(input_current_list[:max_time])
            
        if len(input_current) != 1000:
            print("ERROR")

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
    with open('../data/data_encoding.pkl', 'wb') as handle:
        pkl.dump(encoded_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open('../data/label_encoding.pkl', 'wb') as handle:
        pkl.dump(encoded_label, handle, protocol=pkl.HIGHEST_PROTOCOL)


def fix_time_noisy(temp_jitter=False):
    # import neuron params
    from tactile_encoding.parameters.ideal_params import neuron_parameters, input_currents, time_points, runtime

    classes = neuron_parameters.keys()

    # run the enocding and create training data
    max_trials = 100
    max_time = 1000  # ms
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
            # variable input currents over time
            if len(input_currents[class_name]) > 1:
                if time_points[class_name] is None:
                    print('Missing time points.')
                input_current = np.zeros((sim_time, 1))
                # TODO add temp jitter here and keep repeating signal until 1sec reached
                for counter, actual_current in enumerate(input_currents[class_name]):
                    if counter == 0:
                        input_current[:time_points[class_name]
                                      [counter]] = actual_current

                    elif counter < len(time_points[class_name]):
                        input_current[time_points[class_name][counter-1]
                            :time_points[class_name][counter]] = actual_current

                    elif counter < len(time_points[class_name]):
                        input_current[time_points[class_name]
                                      [counter-1]:] = actual_current
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
                noise = np.random.normal(
                    loc=0.0, scale=0.1, size=input_current.size)
                input_current_list = np.array(
                    [input_current[x] + noise[x] for x in range(len(input_current))])
                for _ in range(factor):
                    noise = np.random.normal(
                        loc=0.0, scale=0.1, size=input_current.size)
                    input_current_list = np.append(input_current_list, np.array(
                        [input_current[x] + noise[x] for x in range(len(input_current))]), axis=0)

                # cut down to fix length
                input_current = np.array(input_current_list[:max_time])

            else:
                noise = np.random.normal(
                    loc=0.0, scale=0.1, size=input_current.size)
                input_current = np.array(
                    [input_current[x] + noise[x] for x in range(len(input_current))])

            if len(input_current) != 1000:
                    print("ERROR")

            input = torch.as_tensor(input_current)

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

    filename = '../data/data_encoding_noisy'

    if temp_jitter:
        filename = filename + '_temp_jitter'

    encoded_data = np.array(encoded_data)
    encoded_label = np.array(encoded_label)

    # dump neuron output to file
    with open(filename, 'wb') as handle:
        pkl.dump(encoded_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open('../data/label_encoding.pkl', 'wb') as handle:
        pkl.dump(encoded_label, handle, protocol=pkl.HIGHEST_PROTOCOL)


def fix_time_noisy_test(temp_jitter=False):
    # import neuron params
    from tactile_encoding.parameters.ideal_params import neuron_parameters, input_currents, time_points, runtime

    classes = neuron_parameters.keys()

    # run the enocding and create training data
    max_trials = 2
    max_time = 1000  # ms
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
            # calc if stacking input is needed
            if sim_time < max_time:
                factor = round((max_time/sim_time)+0.5)
            else:
                factor = 1
                
            for rep_counter in range(factor):
                # variable input current
                if len(input_currents[class_name]) > 1:
                    if time_points[class_name] is None:
                        print('Missing time points.')
                    input_current = np.zeros((sim_time, 1))
                    # TODO add temp jitter here and keep repeating signal until 1sec reached
                    for counter, actual_current in enumerate(input_currents[class_name]):
                        if counter == 0:
                            input_current[:time_points[class_name]
                                        [counter]] = actual_current

                        elif counter < len(time_points[class_name]):
                            input_current[time_points[class_name][counter-1]
                                :time_points[class_name][counter]] = actual_current

                        elif counter < len(time_points[class_name]):
                            input_current[time_points[class_name]
                                        [counter-1]:] = actual_current
                # const input current
                else:

                    input_current = np.ones((max_time, 1)) * \
                        input_currents[class_name]
                    noise = np.random.normal(
                        loc=0.0, scale=0.1, size=input_current.size)
                    input_current = np.array(
                        [input_current[x] + noise[x] for x in range(len(input_current))])
                    break

            # set up MN neuron
            neurons = MN_neuron(
                1, neuron_parameters[class_name], dt=1E-3, train=False)

            if len(input_current) != 1000:
                    print("ERROR")

            input = torch.as_tensor(input_current)

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

    filename = '../data/data_encoding_noisy'

    if temp_jitter:
        filename = filename + '_temp_jitter'

    encoded_data = np.array(encoded_data)
    encoded_label = np.array(encoded_label)

    # dump neuron output to file
    with open(filename, 'wb') as handle:
        pkl.dump(encoded_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open('../data/label_encoding.pkl', 'wb') as handle:
        pkl.dump(encoded_label, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    
    #fix_time_noisy_test()
    
    # original length
    print('Creating original data.')
    original()

    print('Creating noisy original data.')
    original(add_noise=True, temp_jitter=False)

    # TODO implement temporal jitter
    # print('Creating noisy original data with temporal jitter.')
    # original(add_noise=True, temp_jitter=True)

    # fix 1000ms length
    print('Creating 1000ms data.')
    fix_time()

    print('Creating noisy 1000ms data.')
    fix_time_noisy(temp_jitter=False)

    # TODO implement temporal jitter
    # print('Creating noisy 1000ms data with temporal jitter.')
    # fix_time_noisy(add_noise=True, temp_jitter=True)


    print('Finished with data creation.')
