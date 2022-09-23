'''
Code to play around with the parameters of the Mihalas-Nieburg neuron
'''
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import trange
from datasets import load_analog_data
from models import Encoder, MN_neuron
from parameters.MN_params import MNparams_dict, INIT_MODE
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def main():
    device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
            
    upsample_fac = 1
    file_name = "data/data_braille_letters_digits.pkl"
    ds_train, _, _, nb_inputs, nb_steps = load_analog_data(file_name, upsample_fac)
    # Extract data
    # data = data_dict['taxel_data']
    # labels = data_dict['letter']
    # nb_inputs = len(data[0][0])
    # nb_steps = len(data[0])

    # select random trial
    rand_nb = int(random.random()*len(ds_train))

    # create neuron response
    nb_input_copies = 1
    encoder_weight_scale = 1
    fwd_weight_scale = 1E-6
    a = torch.empty((nb_inputs,))
    torch.nn.init.normal_(
        a, mean=MNparams_dict[INIT_MODE][0], std=fwd_weight_scale / np.sqrt(nb_inputs))

    A1 = torch.empty((nb_inputs,))
    torch.nn.init.normal_(
        A1, mean=MNparams_dict[INIT_MODE][1], std=fwd_weight_scale / np.sqrt(nb_inputs))

    A2 = torch.empty((nb_inputs,))
    torch.nn.init.normal_(
        A2, mean=MNparams_dict[INIT_MODE][2], std=fwd_weight_scale / np.sqrt(nb_inputs))

    network = torch.nn.Sequential(Encoder(nb_inputs, encoder_weight_scale, nb_input_copies),
                        MN_neuron(nb_inputs, a, A1, A2, train=True),).to(device)
    print(network)

    # run the model
    writer = SummaryWriter()  # For logging purpose

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    pbar = trange(nb_epochs)

    for e in pbar:
        local_loss = []
        accs = []  # accs: mean training accuracies for each batch
        for x_local, y_local in dl_train:
            x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)
            for t in range(nb_steps):
                out = network(x_local[:, t])
                print("debug")

    # visualize data and neuron
    plt.subplot(2,1,1)
    plt.title(labels[rand_nb])
    plt.plot(data[rand_nb])
    plt.show()
    

if __name__ == "__main__":
    main()