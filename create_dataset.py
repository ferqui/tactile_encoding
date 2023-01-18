import numpy as np
import pandas as pd
import pickle as pkl

import torch
import torch.nn as nn

from utils.models import MN_neuron
# from utils.utils import check_cuda, train_test_validation_split

def main():
    # set up neuron parameters and input current
    class_names = ["class1", "class2"]

    neuron_parameters = []

    class1 = {
        "a": 6142,
        "A1": 432,
    }
    neuron_parameters.append(class1)

    class2 = {
        "A2": 6142,
        "b": 432,
    }
    neuron_parameters.append(class2)

    input_currents = {
        "class1": 1,
        "class2": 1.3,
    }

    # run the enocding and create training data
    max_trials = 1000
    time_steps = 100
    encoded_data = []
    encoded_label = []
    # create training dataset by iterating over neuron params and input currents
    for counter, params in enumerate(neuron_parameters):
        # extract class name and get the correct input current
        class_name = class_names[counter]
        # build the different input current profiles
        if class_name == "class1":
            current = input_currents[class_name]
            input_current = np.ones((time_steps, 1)) * current
        elif class_name == "class2":
            current = input_currents[class_name]
            input_current = np.ones((time_steps, 1)) * current
        else:
            current = input_currents[class_name]
            input_current = np.ones((time_steps, 1)) * current
        # neuron_parameter = params[1]
        current = input_currents[class_name]
        input_current = np.ones((time_steps, 1)) * current
        # set up MN neuron
        neurons = MN_neuron(1, params, dt=1E-3, train=False)

        # compute neuron output
        input = torch.as_tensor(input_current)
        output_s = []
        print(input.shape[0])
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
