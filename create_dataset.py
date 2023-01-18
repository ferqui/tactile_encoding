import numpy as np
import pandas as pd
import pickle as pkl

import torch
import torch.nn as nn

from utils.models import MN_neuron
# from utils.utils import check_cuda, train_test_validation_split


def main():
    # import neuron params
    from ideal_params import neuron_parameters, input_currents, time_points

    classes = neuron_parameters.keys()
    # run the enocding and create training data
    max_trials = 1000
    time_steps = 100
    encoded_data = []
    encoded_label = []
    # create training dataset by iterating over neuron params and input currents
    for _, class_name in enumerate(classes):
        # iterate over changes
        if class_name == 'Spike latency':
            max_time = 50
        elif class_name == 'Tonic spiking' or class_name == 'Spike frequency adaptation' or class_name == 'Afterpotentials' or class_name == 'Basal bistability':
            max_time = 200
        elif class_name == 'Class 2':
            max_time = 300
        elif class_name == 'Threshold variability' or class_name == 'Integrator' or class_name == 'Hyperpolarizing spiking' or class_name == 'Hyperpolarizing bursting':
            max_time = 400
        elif class_name == 'Class 1' or class_name == 'Phasic spiking' or class_name == 'Tonic bursting' or class_name == 'Phasic bursting' or class_name == 'Mixed mode':
            max_time = 500
        elif class_name == 'Preferred frequency':
            max_time = 800
        elif class_name == 'Accommodation' or class_name == 'Rebound spike' or class_name == 'Input bistability' or class_name == 'Rebound burst':
            max_time = 1000
        else:
            print('No correct time found')
        if len(input_currents[class_name]) > 1:
            current = np.zeros((max_time, 1))
            for counter, actual_current in enumerate(input_currents[class_name]):
                if counter == 0:
                    current[:time_points[class_name][counter]] = actual_current
                elif counter < len(time_points[class_name]):
                    current[time_points[class_name][counter-1]:time_points[class_name][counter]] = actual_current
                elif counter < len(time_points[class_name]):
                    current[time_points[class_name]
                            [counter-1]:] = actual_current
        else:
            # const current
            input_current = np.ones((time_steps, 1)) * \
                input_currents[class_name]
        # set up MN neuron
        neurons = MN_neuron(
            1, neuron_parameters[class_name], dt=1E-3, train=False)

        # compute neuron output
        input = torch.as_tensor(input_current)
        output_s = []
        for t in range(input.shape[0]):
            out = neurons(input[t])
            output_s.append(out.cpu().numpy())
        output_s = np.stack(output_s)

        # create max_trials trials per class
        for _ in range(max_trials):
            # store neuron output
            encoded_data.append(output_s)
            encoded_label.append(class_name)

    # TODO dump neuron output to file
    with open('./data_encoding', 'wb') as handle:
        pkl.dump(encoded_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open('./label_encoding', 'wb') as handle:
        pkl.dump(encoded_label, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
