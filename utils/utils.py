import numpy as np
import os
import torch

from sklearn.model_selection import train_test_split


def check_cuda():
    # check for available GPU and distribute work
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
    return device


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


def calc_sigma_linear(generations, generation):
    """
    Returns a line with negative slope.
    The function start at 1, hits >=0.5 at generations/2 and approaches 0.01 at generations.
    """

    sigma_start = 1
    sigma_stop = 0.01
    sigma = ((sigma_stop-sigma_start)/generations)*generation+sigma_start
    return sigma


def calc_sigma_sigmoid(generations, generation):
    """
    Returns a invert sigmoid function invariant to change of generations.
    The function start at 1, hits 0.5 at generations/2 and approaches 0 at generations.
    """

    stretch = 12  # the lower, the more linear, the higher the closer to step function at generation/2
    fac = stretch/generations
    return (1-(1/(1+np.exp(-(generation*fac)+(stretch/2)))))