import os
import torch
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.model_selection import train_test_split


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


def check_cuda(share_GPU=False, gpu_sel=0, gpu_mem_frac=0.5):
    """Check for available GPU and distribute work (if needed/wanted)"""

    if (torch.cuda.device_count()>1) & (share_GPU):
        gpu_av = [torch.cuda.is_available() for ii in range(torch.cuda.device_count())]
        print("Detected {} GPUs. The load will be shared.".format(torch.cuda.device_count()))
        for gpu in range(len(gpu_av)):
            if True in gpu_av:
                if gpu_av[gpu_sel]:
                    device = torch.device("cuda:"+str(gpu))
                    print("Selected GPUs: {}" .format("cuda:"+str(gpu)))
                else:
                    device = torch.device("cuda:"+str(gpu_av.index(True)))
            else:
                device = torch.device("cpu")
                print("No available GPU detected. Running on CPU.")
    elif (torch.cuda.device_count()>1) & (not share_GPU):
        print("Multiple GPUs detected but single GPU selected. Setting up the simulation on {}".format("cuda:"+str(gpu_sel)))
        device = torch.device("cuda:"+str(gpu_sel))
        torch.cuda.set_per_process_memory_fraction(gpu_mem_frac, device=device) # decrese or comment out memory fraction if more is available (the smaller the better)
    else:
        if torch.cuda.is_available():
            print("Single GPU detected. Setting up the simulation there.")
            device = torch.device("cuda")
            torch.cuda.set_per_process_memory_fraction(gpu_mem_frac) # decrese or comment out memory fraction if more is available (the smaller the better)
        else:
            device = torch.device("cpu")
            print("No GPU detected. Running on CPU.")

    """
    # Simon's version for 'default' load distribution
    if torch.cuda.device_count() > 1:
        torch.cuda.empty_cache()
        gpu_sel = 1
        gpu_av = [torch.cuda.is_available()
                for ii in range(torch.cuda.device_count())]
        print("Detected {} GPUs. The load will be shared.".format(
            torch.cuda.device_count()))
        for gpu in range(len(gpu_av)):
            if True in gpu_av:
                if gpu_av[gpu_sel]:
                    device = torch.device("cuda:"+str(gpu))
                    # torch.cuda.set_per_process_memory_fraction(0.9, device=device)
                    print("Selected GPUs: {}" .format("cuda:"+str(gpu)))
                else:
                    device = torch.device("cuda:"+str(gpu_av.index(True)))
            else:
                device = torch.device("cpu")
                print("No GPU detected. Running on CPU.")
    else:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            print("Single GPU detected. Setting up the simulation there.")
            device = torch.device("cuda:0")
            # torch.cuda.set_per_process_memory_fraction(0.9, device=device)
        else:
            device = torch.device("cpu")
            print("No GPU detected. Running on CPU.")
    """
    
    return device


def train_test_validation_split(data, label, split=[70, 20, 10], seed=None):
    """
    Creates a train-test-validation split using the sklearn train_test_split() twice.
    Function accepts lists, arrays, and tensor.
    Default split: [70, 20, 10]

    data.shape: [trials, time, sensor]
    label.shape: [trials] 
    split: [train, test, validation]
    """
    # do some sanity checks first
    if len(split) != 3:
        raise ValueError(
            f"Split dimensions are wrong. Expected 3 , but got {len(split)}. Please provide split in the form [train size, test size, validation size].")
    if min(split) == 0.0:
        raise ValueError(
            "Found entry 0.0. If you want to use only perfrom a two-folded split, use the sklearn train_test_split function only please.")
    if sum(split) > 99.0:
        split = [x/100 for x in split]
    if sum(split) < 0.99:
        raise ValueError("Please use a split summing up to 1, or 100%.")

    # create train and (test + validation) split
    x_test_validation, x_train, y_test_validation, y_train = train_test_split(
        data, label, test_size=split[0], shuffle=True, stratify=label, random_state=seed)
    # create test and validation split
    ratio = split[1]/sum(split[1:])
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_test_validation, y_test_validation, test_size=ratio, shuffle=True, stratify=y_test_validation, random_state=seed)

    return x_train, y_train, x_test, y_test, x_validation, y_validation


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


def load_layers(layers, map_location, variable=False, requires_grad=True):
    
    if variable: # meaning that the weights are not to be loaded <-- layers is a variable name
        
        lays = layers
        
        for ii in lays:
            ii.to(map_location)
            ii.requires_grad = requires_grad
    
    else: # meaning that weights are to be loaded from a file <-- layers is a path
        
        lays = torch.load(layers, map_location=map_location)
        
        for ii in lays:
            ii.requires_grad = requires_grad
        
    return lays


def retrieve_nni_results(
    exp_name,
    exp_id,
    metrics,
    max_trial_num=10000,
    nni_default_path=True,
    export_csv=False
    ):
    """
    Given an NNI experiment, it returns the trial ID with the highest value of metrics (and the value as well).

    Fra, Vittorio; Politecnico di Torino; EDA Group; Torino, Italy.
    Klepatsch, Daniel; Silicon Austria Labs; Graz, Austria.
    """

    if nni_default_path:
        db_path = os.path.expanduser("~/nni-experiments/{}/{}/db".format(exp_name,exp_id))
    else:
        db_path = nni_default_path
    
    con = sqlite3.connect(os.path.join(db_path,"nni.sqlite")) # sqlite connector

    # Load the data into a DataFrame
    trial_data = pd.read_sql_query(" select m.timestamp, t.trialJobId, t.data as params, m.type, m.data as results  "
                                   " from TrialJobEvent as t INNER JOIN MetricData as m ON t.trialJobId = m.trialJobId "
                                   " where m.type == \"FINAL\" and t.event == \"WAITING\" and t.sequenceId <= (?) ;", con, params=(max_trial_num,))
    
    # Process top10 results of default = test accuracy
    results_data = pd.json_normalize(trial_data['results'].map(eval).map(eval))
    params_data = pd.json_normalize(trial_data['params'].map(eval))

    df_trial = pd.concat([trial_data.drop(['results','params'], axis=1), params_data, results_data], axis=1)
    
    top = df_trial.sort_values(by=[metrics,"timestamp"], ascending=False)#.head(10)
    
    if export_csv:
        export_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fileNamecsv = 'OptimizationResults_{}_{}_{}.csv'.format(
            exp_name,
            exp_id,
            export_datetime)
        top.to_csv(fileNamecsv, index=False)

    con.close()

    return top.iloc[0]["trialJobId"], top.iloc[0][metrics]

