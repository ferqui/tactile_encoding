import os
import numpy as np


def addHeaderToMetadata(filename, header):
    file = os.path.join(filename)

    f = open(file, 'a')

    f.write('\n--------------------- ' + header + ' ---------------------\n')

    f.close()


def addToNetMetadata(filename, key, value, header=''):
    file = os.path.join(filename)

    f = open(file, 'a')

    if header != '':
        f.write('\n--------------------- ' + header + ' ---------------------\n')

    f.write(key + ':' + ' ' + str(value) + '\n')

    f.close()


def set_results_folder(depth, result_dir='experiments/results/'):
    """
    Prepare path to experiment folder.
    """

    if not (os.path.isdir(result_dir)):
        os.mkdir(result_dir)
    sim_dir = result_dir
    for level in depth:
        sim_dir = os.path.join(
            sim_dir,
            level)
        if not (os.path.isdir(sim_dir)):
            os.mkdir(sim_dir)
    return sim_dir


def generate_dict(key_to_change, variable_range, force_param_dict):
    file_dir_params = 'parameters/'
    param_filename = 'parameters_mn'
    file_name_parameters = file_dir_params + param_filename + '.txt'
    params = {}
    num_values = len(variable_range)

    with open(file_name_parameters) as file:
        for line in file:
            (key, value) = line.split()
            if key == key_to_change:
                params[key] = variable_range
            else:

                if key in force_param_dict.keys():
                    value = force_param_dict[key]
                params[key] = np.linspace(float(value), float(value), num_values)

    return params
