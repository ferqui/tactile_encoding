import numpy as np
import pandas as pd
import pickle as pkl

import torch
import torch.nn as nn

from utils.models import MN_neuron
from utils.utils import check_cuda, train_test_validation_split


def main():
    device = check_cuda()
    # import neuron params
    from ideal_params import neuron_parameters, input_currents, time_points, runtime

    classes = neuron_parameters.keys()
    # run the enocding and create training data
    max_trials = 1000
    max_time = 1000 # ms
    encoded_data_original = []
    encoded_data = []
    encoded_label = []
    # create training dataset by iterating over neuron params and input currents
    for _, class_name in enumerate(classes):
        # iterate over changes
        sim_time = runtime[class_name]
        if len(input_currents[class_name]) > 1:
            current = np.zeros((sim_time, 1))
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
            input_current = np.ones((sim_time, 1)) * \
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
        encoded_data_original.append(output_s)

        # stack input current trace if input length < 1000ms
        factor = int((max_time/sim_time)+0.5)
        if factor > 1:
            for clone in range(factor):
                input_current = np.append(input_current, input_current)
            input_current = input_current[:max_time]
            input = torch.as_tensor(input_current)

        # compute new neuron output
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

    # dump neuron output to file
    with open('./data_encoding_original', 'wb') as handle:
        pkl.dump(encoded_data_original, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open('./data_encoding', 'wb') as handle:
        pkl.dump(encoded_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open('./label_encoding', 'wb') as handle:
        pkl.dump(encoded_label, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
