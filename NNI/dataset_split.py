import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pickle as pkl
from torch.utils.data import TensorDataset

from utils.ideal_params import input_currents


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile! :(
            return None
        return directory_path


def train_validation_test_split(data, label, split=[70, 20, 10], seed=None, multiple=False, save_dataset=False, save_tensor=False, labels_mapping=None, save_name=None, save_path=None):
    """
    Creates train-validation-test splits using the sklearn train_test_split() twice.
    Can be used either to prepare "ready-to-use" splits or to create and store splits.

    If multiple splits are not needed and no saving option is set, the lists x_train, y_train, x_val, y_val, x_test, y_test are returned (without labels mapping).

    Function accepts lists, arrays, and tensor.
    Default split: [70, 20, 10]

    data.shape: [trials, time, sensor]
    label.shape: [trials] 
    split: [train, test, validation]
    """

    if multiple:
        if (not save_dataset) & (not save_tensor):
            raise ValueError("Multiple train-val splits are created but no saving option is enabled.")

    if save_dataset | save_tensor:
        if (save_path == None) | (save_name == None):
            raise ValueError("Check a file name and a path are provided to save the datasets.")
        create_directory(save_path)
    
    filename_prefix = save_path + save_name

    # do some sanity checks first
    if len(split) != 3:
        raise ValueError(
            f"Split dimensions are wrong. Expected 3 but got {len(split)}. Please provide split in the form [train size, test size, validation size].")
    if min(split) == 0.0:
        raise ValueError(
            "Found entry 0.0. If you want to use only perfrom a two-folded split, use the sklearn train_test_split function only please.")
    if sum(split) > 99.0:
        split = [x/100 for x in split]
    if sum(split) < 0.99:
        raise ValueError("Please use a split summing up to 1, or 100%.")
    
    train, val, test = split
    split_1 = test
    split_2 = 1 - train/(train+val)

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        data, label, test_size=split_1, shuffle=True, stratify=label, random_state=seed)
    
    
    if save_dataset: # Save the test split
        filename_test = filename_prefix + "_test"
        # xs test
        with open(f"{filename_test}.pkl", 'wb') as handle:
            pkl.dump(np.array(x_test, dtype=object), handle,
                        protocol=pkl.HIGHEST_PROTOCOL)
        # ys test
        with open(f"{filename_test}_label.pkl", 'wb') as handle:
            pkl.dump(np.array(y_test, dtype=object), handle,
                        protocol=pkl.HIGHEST_PROTOCOL)
    
    if save_tensor:
        filename_test = filename_prefix + "_ds_test"
        x_test = torch.as_tensor(np.array(x_test), dtype=torch.float)
        labels_test = torch.as_tensor(value2index(
            y_test, labels_mapping), dtype=torch.long)
        ds_test = TensorDataset(x_test, labels_test)
        torch.save(ds_test, "{}.pt".format(filename_test))
    
    if multiple:

        for ii in range(10):

            x_train, x_val, y_train, y_val = train_test_split(
                x_trainval, y_trainval, test_size=split_2, shuffle=True, stratify=y_trainval, random_state=seed)
            
            if save_dataset:

                filename_train = filename_prefix + "_train"
                filename_val = filename_prefix + "_val"

                # xs training
                with open(f"{filename_train}_{ii}.pkl", 'wb') as handle:
                    pkl.dump(np.array(x_train, dtype=object), handle,
                                protocol=pkl.HIGHEST_PROTOCOL)
                # ys training
                with open(f"{filename_train}_{ii}_label.pkl", 'wb') as handle:
                    pkl.dump(np.array(y_train, dtype=object), handle,
                                protocol=pkl.HIGHEST_PROTOCOL)
                
                # xs validation
                with open(f"{filename_val}_{ii}.pkl", 'wb') as handle:
                    pkl.dump(np.array(x_val, dtype=object), handle,
                                protocol=pkl.HIGHEST_PROTOCOL)
                # ys validation
                with open(f"{filename_val}_{ii}_label.pkl", 'wb') as handle:
                    pkl.dump(np.array(y_val, dtype=object), handle,
                                protocol=pkl.HIGHEST_PROTOCOL)

            if save_tensor:

                filename_train = filename_prefix + "_ds_train"
                filename_val = filename_prefix + "_ds_val"

                x_train = torch.as_tensor(np.array(x_train), dtype=torch.float)
                labels_train = torch.as_tensor(value2index(
                    y_train, labels_mapping), dtype=torch.long)
        
                x_validation = torch.as_tensor(
                    np.array(x_val), dtype=torch.float)
                labels_validation = torch.as_tensor(value2index(
                    y_val, labels_mapping), dtype=torch.long)
                
                ds_train = TensorDataset(x_train, labels_train)
                ds_val = TensorDataset(x_validation, labels_validation)
                
                torch.save(ds_train, "{}_{}.pt".format(filename_train,ii))
                torch.save(ds_val, "{}_{}.pt".format(filename_val,ii))

    else:

        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval, test_size=split_2, shuffle=True, stratify=y_trainval, random_state=seed)
        
        if save_dataset:

            filename_train = filename_prefix + "_train"
            filename_val = filename_prefix + "_val"
        
            # xs training
            with open(f"{filename_train}.pkl", 'wb') as handle:
                pkl.dump(np.array(x_train, dtype=object), handle,
                            protocol=pkl.HIGHEST_PROTOCOL)
            # ys training
            with open(f"{filename_train}_label.pkl", 'wb') as handle:
                pkl.dump(np.array(y_train, dtype=object), handle,
                            protocol=pkl.HIGHEST_PROTOCOL)

            # xs validation
            with open(f"{filename_val}.pkl", 'wb') as handle:
                pkl.dump(np.array(x_val, dtype=object), handle,
                            protocol=pkl.HIGHEST_PROTOCOL)
            # ys validation
            with open(f"{filename_val}_label.pkl", 'wb') as handle:
                pkl.dump(np.array(y_val, dtype=object), handle,
                            protocol=pkl.HIGHEST_PROTOCOL)

        if save_tensor:

            filename_train = filename_prefix + "_ds_train"
            filename_val = filename_prefix + "_ds_val"

            x_train = torch.as_tensor(np.array(x_train), dtype=torch.float)
            labels_train = torch.as_tensor(value2index(
                y_train, labels_mapping), dtype=torch.long)
        
            x_validation = torch.as_tensor(
                np.array(x_val), dtype=torch.float)
            labels_validation = torch.as_tensor(value2index(
                y_val, labels_mapping), dtype=torch.long)
                
            ds_train = TensorDataset(x_train, labels_train)
            ds_val = TensorDataset(x_validation, labels_validation)
                
            torch.save(ds_train, filename_train)
            torch.save(ds_val, filename_val)

        return x_train, y_train, x_val, y_val, x_test, y_test


def value2key(entry, dictionary):
    if (type(entry) != list) & (type(entry) != np.ndarray):

        key = [list(dictionary.keys())[list(dictionary.values()).index(entry)]]
    
    else:

        key = [list(dictionary.keys())[list(dictionary.values()).index(e)] for e in entry]
        
    return key


def value2index(entry, dictionary):
    if (type(entry) != list) & (type(entry) != np.ndarray):

        idx = [list(dictionary.values()).index(entry)]
        
    else:

        idx = [list(dictionary.values()).index(e) for e in entry]
        
    return idx



# Specify what kind of data to take into account
original = True
fixed_length = not original
noise = False
jitter = False

# prepare data selection
data_filepath = "../data/data_encoding"
label_filepath = "../data/label_encoding"
name = ""
data_features = [original, fixed_length, noise, jitter]
data_attributes = ["original", "fix_len", "noisy", "temp_jitter"]
for num,el in enumerate(list(np.where(np.array(data_features)==True)[0])):
    data_filepath += "_{}".format(data_attributes[el])
    label_filepath += "_{}".format(data_attributes[el])
    name += "{} ".format(data_attributes[el])
data_filepath += ".pkl"
label_filepath += ".pkl"
name = name[:-1]

infile = open(data_filepath, "rb")
encoded_data = pkl.load(infile)
infile.close()

if original:
    n_copies = 100
    original_panels = []
    for num_panel,el_panel in enumerate(encoded_data):
        original_variables = []
        for num_variable,el_variable in enumerate(el_panel):
            original_signal = []
            if num_variable < 2:
                for num_signal,el_signal in enumerate(el_variable):
                    original_signal.append([el_signal.item()])
            else:
                for num_signal,el_signal in enumerate(el_variable):
                    original_signal.append(el_signal)
            original_variables.append(original_signal)
        original_panels.append(np.array(original_variables))
    original_labels = list(input_currents.keys())
    original_data_extended = []
    original_labels_extended = []
    for num,el in enumerate(original_panels):
        for ii in range(n_copies):
            original_data_extended.append(el)
            original_labels_extended.append(original_labels[num])
    encoded_data = original_data_extended
    encoded_label = original_labels_extended
else:
    infile = open(label_filepath, "rb")
    encoded_label = pkl.load(infile)
    infile.close()

labels_MNpaper = {
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

if original:
    ### Save the extended version (with n_copies replicas) of the original data
    filename_data_extended = "../data/data_encoding_{}".format(name.replace(" ","_")) + "_extended"
    filename_label_extended = "../data/label_encoding_{}".format(name.replace(" ","_")) + "_extended"
    # xs
    with open(f"{filename_data_extended}.pkl", 'wb') as handle:
        pkl.dump(encoded_data, handle,
                    protocol=pkl.HIGHEST_PROTOCOL)
    # ys
    with open(f"{filename_label_extended}.pkl", 'wb') as handle:
        pkl.dump(value2index(encoded_label, labels_MNpaper), handle,
                    protocol=pkl.HIGHEST_PROTOCOL)
    ### Save the splits
    save_path = "../dataset_splits/{}_extended/".format(name.replace(" ","_"))
    create_directory(save_path)
    filename_prefix = save_path + name.replace(" ","_") + "_extended"
    filename_train = filename_prefix + "_ds_train"
    filename_val = filename_prefix + "_ds_val"
    filename_test = filename_prefix + "_ds_test"
    # xs training
    xs_training = []
    for ii in range(len(original_panels)):
        xs_training.extend(encoded_data[ii*n_copies:70+ii*n_copies])
    with open(f"{filename_train}.pkl", 'wb') as handle:
        pkl.dump(xs_training, handle,
                    protocol=pkl.HIGHEST_PROTOCOL)
    # ys training
    ys_training = []
    for ii in range(len(original_panels)):
        ys_training.extend(value2index(encoded_label[ii*n_copies:70+ii*n_copies], labels_MNpaper))
    with open(f"{filename_train}_label.pkl", 'wb') as handle:
        pkl.dump(ys_training, handle,
                    protocol=pkl.HIGHEST_PROTOCOL)
    # xs validation
    xs_validation = []
    for ii in range(len(original_panels)):
        xs_validation.extend(encoded_data[70+ii*n_copies:90+ii*n_copies])
    with open(f"{filename_val}.pkl", 'wb') as handle:
        pkl.dump(xs_validation, handle,
                    protocol=pkl.HIGHEST_PROTOCOL)
    # ys validation
    ys_validation = []
    for ii in range(len(original_panels)):
        ys_validation.extend(value2index(encoded_label[70+ii*n_copies:90+ii*n_copies], labels_MNpaper))
    with open(f"{filename_val}_label.pkl", 'wb') as handle:
        pkl.dump(ys_validation, handle,
                    protocol=pkl.HIGHEST_PROTOCOL)
    # xs test
    xs_test = []
    for ii in range(len(original_panels)):
        xs_test.extend(encoded_data[90+ii*n_copies:(ii+1)*n_copies])
    with open(f"{filename_test}.pkl", 'wb') as handle:
        pkl.dump(xs_test, handle,
                    protocol=pkl.HIGHEST_PROTOCOL)
    # ys test
    ys_test = []
    for ii in range(len(original_panels)):
        ys_test.extend(value2index(encoded_label[90+ii*n_copies:(ii+1)*n_copies], labels_MNpaper))
    with open(f"{filename_test}_label.pkl", 'wb') as handle:
        pkl.dump(ys_test, handle,
                    protocol=pkl.HIGHEST_PROTOCOL)
else:
    train_validation_test_split(np.array(encoded_data)[:, 0], encoded_label, 
                                multiple=True, 
                                save_tensor=True,
                                labels_mapping=labels_MNpaper, 
                                save_name=name.replace(" ","_"),
                                save_path="../dataset_splits/{}/".format(name.replace(" ","_")))