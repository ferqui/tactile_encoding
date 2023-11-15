import nni
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import json
from pathlib import Path
import pickle

# import torchviz
import matplotlib.pyplot as plt

from tqdm import trange

import numpy as np
import h5py

from datasets import load_data

from parameters.MN_params import MNparams_dict, INIT_MODE
from models import Encoder, LIF_neuron, MN_neuron_sp, ALIF_neuron
from auxiliary import set_random_seed

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
class MNISTDataset_current(torch.utils.data.dataset.Dataset):
    """
    Load MNIST dataset as custom dataset generated by converting MNIST digits into input currents or into feature sets
    (frequency values, amplitudes and slopes).
    """

    def __init__(self, hdf5_file, device=None):
        """
            :param data_file: Path to h5 file with dataset.
        """
        self.file = hdf5_file
        self.device = device

        for attr in self.file.attrs.keys():
            eval_str = "self.{}".format(attr) + " = " + str(self.file.attrs[attr])
            exec(eval_str)

    def __getitem__(self, idx):
        values = self.file['values'][idx]
        target = self.file['targets'][idx]
        idx_time = self.file['idx_time'][idx]
        idx_inputs = self.file['idx_inputs'][idx]
        idx = np.vstack((idx_time, idx_inputs))
        # From sparse to dense
        data = torch.sparse_coo_tensor(idx, values, (self.n_time_steps, self.n_inputs)).to_dense()

        return data, target

    def __len__(self):
        return len(self.file['targets'])
parameters_thenc = {}
with open("parameters/parameters_thenc.txt") as f:
    for line in f:
        (key, val) = line.split()
        parameters_thenc[key] = val

firing_mode_dict = {
    "FA": {"a": 5, "A1": 0, "A2": 0},
    "SA": {"a": 0, "A1": 0, "A2": 0},
    "MIX": {"a": 5, "A1": 5, "A2": -0.3},
}

MN_dict_param = {
    "a": {"ini": 5, "train": True, "custom_lr": 5e-3},
    "A1": {"ini": 0, "train": True, "custom_lr": 5e-3},
    "A2": {"ini": 0, "train": True, "custom_lr": 5e-3},
    "b": {"ini": 10, "train": True, "custom_lr": 5e-3},
    "G": {"ini": 50, "train": True, "custom_lr": 5e-3},
    "k1": {"ini": 200, "train": False, "custom_lr": None},
    "k2": {"ini": 20, "train": False, "custom_lr": None},
    "R1": {"ini": 0, "train": True, "custom_lr": 5e-3},
    "R2": {"ini": 1, "train": True, "custom_lr": 5e-3},
}

ALIF_dict_param = {
    "alpha": {"ini": 1, "train": False, "custom_lr": None},
    "beta_alif": {"ini": 1, "train": True, "custom_lr": None},
    "b_0": {"ini": 0.1, "train": True, "custom_lr": None},
    "tau_adp": {"ini": 1, "train": True, "custom_lr": None},
    "beta_adapt": {"ini": 1.8, "train": True, "custom_lr": None},
}


def training(x_local,y_local,device,network,log_softmax_fn,loss_fn,optimizer,args,dict_param,time = None):
        #pbar.set_description(f"{batch_idx}/{len(dl_train)}")
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(
            device, non_blocking=True
        )
        y_local = y_local.long()
        # Reset all the layers in the network
        for layer in network:
            if hasattr(layer.__class__, "reset"):
                layer.reset()

        # Simulate the network
        # we are going to record the hidden layer
        # spikes for regularization purposes
        loss_local = 0

        l0_spk = []
        lif1_spk = []
        lif2_spk = []

        l0_mem = []
        lif1_mem = []
        lif2_mem = []
        time.reset(total=x_local.shape[1])
        if not args.fast:
            recorder = {'V':[],'i1':[],'i2':[],'Thr':[]}
            for t in range(x_local.shape[1]):
               out = network(x_local[:, t]*args.gain)
               l0_spk.append(network[1].state.spk)
               l0_mem.append(network[1].state.V)
               lif1_spk.append(network[2].state.S)
               lif1_mem.append(network[2].state.mem)
               lif2_spk.append(network[3].state.S)
               lif2_mem.append(network[3].state.mem)
               recorder['V'].append(network[1].state.V)
               recorder['i1'].append(network[1].state.i1)
               recorder['i2'].append(network[1].state.i2)
               recorder['Thr'].append(network[1].state.Thr)
               time.update()
        else:
            recorder = None
            for t in range(x_local.shape[1]):
                out = network(x_local[:, t]*args.gain)
                lif1_spk.append(network[2].state.S)
                lif2_spk.append(network[3].state.S)
                time.update()

        if not args.fast:
                l0_spk = torch.stack(l0_spk, dim=1)
                l0_mem = torch.stack(l0_mem, dim=1)
        lif1_spk = torch.stack(lif1_spk, dim=1)
        # l1_events = np.where(lif1_spk[0, :, :].cpu().detach().numpy())
        # plt.figure()
        # plt.scatter(l1_events[0], l1_events[1], s=0.1)
        if not args.fast:
                lif1_mem = torch.stack(lif1_mem, dim=1)
        lif2_spk = torch.stack(lif2_spk, dim=1)
        # plt.figure()
        # l2_events = np.where(lif2_spk[0, :, :].cpu().detach().numpy())
        # plt.scatter(l2_events[0], l2_events[1], s=0.1)
        # plt.show()
        if not args.fast:
                lif2_mem = torch.stack(lif2_mem, dim=1)
        m = torch.sum(lif2_spk, 1)  # sum over time
        # print('lif2_sum',m)

        log_p_y = log_softmax_fn(m)

        # Here we can set up our regularizer loss
        reg_loss = args.reg_spikes_l1 * torch.mean(
            torch.sum(lif1_spk, 1)
        )  # e.g., L1 loss on total number of spikes (original: 1e-3)
        reg_loss += args.reg_neurons_l1 * torch.mean(
            torch.sum(torch.sum(lif1_spk, dim=0), dim=0) ** 2
        )  # L1 loss on spikes per neuron (original: 2e-6)
        reg_loss += args.reg_spikes_l2 * torch.mean(
            torch.sum(lif2_spk, 1)
        )  # L2 loss on output layer spikes (original: 1e-3)
        reg_loss += args.reg_neurons_l2 * torch.mean(
            torch.sum(torch.sum(lif2_spk, dim=0), dim=0) ** 2
        )  # L2 loss on output layer spikes per neuron (original: 2e-6)

        # Here we combine supervised loss and the regularizer
        loss_val = loss_fn(log_p_y, y_local) + reg_loss
        optimizer.zero_grad()
        # loss_val.backward()

        loss_val.backward(create_graph=args.gr>=0)  # backpropagation of original loss
        grad_dict = {}
        if args.gr >= 0:
            for param in dict_param:
                if dict_param[param]["param"].grad is not None:
                    grad_dict[param+'b4gr'] = dict_param[param]["param"].grad.clone()
            loss_DB = args.gr * sum(
                [
                    torch.abs(kv[1]["param"].grad).sum()
                    for kv in filter(lambda kv: kv[1]["train"], dict_param.items())
                ]
            )  # computing GR term

            loss_DB.backward()  # backpropagation of GR ter

        else:
            loss_DB = torch.tensor(0)
        optimizer.step()
        for param in dict_param:
            if dict_param[param]["param"].grad is not None:
                grad_dict[param] = dict_param[param]["param"].grad
                dict_param[param]["param"].grad = None
        with torch.no_grad():
            # compare to labels
            _, am = torch.max(m, 1)  # argmax over output units
            accuracy = np.mean((y_local == am).detach().cpu().numpy())
            #accs.append(tmp)
        return loss_val.item(),accuracy,loss_DB.item(),grad_dict,m,recorder

@torch.no_grad()
def compute_classification_accuracy(dataset, network, early, device, args,fast=True, batch=None, time=None):
    accs = []
    multi_accs = []
    ttc = None
    if batch is None:
        pass
    else:
        batch.set_description('Testing')
        batch.reset(total=len(dataset))
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(
            device, non_blocking=True
        )
        y_local = y_local[:, 0]

        for layer in network:
            if hasattr(layer.__class__, "reset"):
                layer.reset()

        mn_spk = []
        lif1_spk = []
        lif2_spk = []

        mn_mem = []
        lif1_mem = []
        lif2_mem = []
        if time is None:
            pass
        else:
            time.reset(total=x_local.shape[1])
        for t in range(x_local.shape[1]):
            out = network(x_local[:, t]*args.gain)

            # Get the spikes and voltages from the MN neuron encoder
            if not fast:
                mn_spk.append(network[1].state.spk)

                mn_mem.append(network[1].state.V)

            # Get the spikes and voltages from the first LIF

            if not fast:
                lif1_spk.append(network[2].state.S)
                lif1_mem.append(network[2].state.mem)

            # Get the spikes and voltages from the second LIF
            lif2_spk.append(network[3].state.S.to_sparse())

            if not fast:
                lif2_mem.append(network[3].state.mem)
            else:
                if t == 0:
                    lif2_sum = network[3].state.S
                else:
                    lif2_sum += network[3].state.S
            if time is not None:
                time.update()
        if not fast:
            mn_spk = torch.stack(mn_spk, dim=1)
            mn_mem = torch.stack(mn_mem, dim=1)
        if not fast:
            lif1_spk = torch.stack(lif1_spk, dim=1)
            lif1_mem = torch.stack(lif1_mem, dim=1)
        lif2_spk = torch.stack(lif2_spk, dim=1).to_dense()

        if not fast:
            lif2_mem = torch.stack(lif2_mem, dim=1)
            lif2_sum = torch.sum(lif2_spk, 1)  # sum over time

        # with output spikes
        _, am = torch.max(lif2_sum, 1)  # argmax over output units
        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)

        if early:
            accs_early = []
            for t in range(lif2_spk.shape[1] - 1):
                # with spiking output layer
                m_early = torch.sum(lif2_spk[:, : t + 1, :], 1)  # sum over time
                _, am_early = torch.max(m_early, 1)  # argmax over output units
                # compare to labels
                tmp_early = np.mean((y_local == am_early).detach().cpu().numpy())
                accs_early.append(tmp_early)
            multi_accs.append(accs_early)
        if batch is not None:
            batch.update()
    if early:
        max_time = int(54 * 25)  # ms
        time_bin_size = int(1)  # ms
        time = range(0, max_time, time_bin_size)

        mean_multi = np.mean(multi_accs, axis=0)
        if np.max(mean_multi) > mean_multi[-1]:
            if mean_multi[-2] == mean_multi[-1]:
                flattening = []
                for ii in range(len(mean_multi) - 2, 1, -1):
                    if mean_multi[ii] != mean_multi[ii - 1]:
                        flattening.append(ii)
                # time to classify
                try:
                    ttc = time[flattening[0]]
                except:
                    ttc = time[-1]
            else:
                # time to classify
                ttc = time[-1]
        else:
            # time to classify
            ttc = time[np.argmax(mean_multi)]

    return np.mean(accs), ttc, mn_spk, lif1_spk, lif2_spk, mn_mem, lif1_mem, lif2_mem
def main(args):
    device = torch.device("cuda:0") if (torch.cuda.is_available() & args.gpu) else torch.device("cpu")
    print(device)

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    ###########################################
    ##                Dataset                ##
    ###########################################
    upsample_fac = 1
    dt = (1 / 100.0) / upsample_fac
    # file_name = "data/data_braille_letters_all.pkl"
    # dataMN = MNIST(
    #     root="data",
    #     train=True,
    #     download=True,
    #     )
    dict_dataset = {}
    batch_size = args.batch_size
    seed = args.seed
    generator = set_random_seed(seed, add_generator=True, device='cpu')
    path_to_dataset = os.path.join(os.getcwd(), 'data','MNIST_time_dataloader')
    train_dataset = MNISTDataset_current(h5py.File(os.path.join(path_to_dataset,'train.h5'), mode='r'), device='cpu')
    test_dataset = MNISTDataset_current(h5py.File(os.path.join(path_to_dataset,'test.h5'), mode='r'), device='cpu')

    dict_dataset['train_loader'] = DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              generator=generator,
                                              num_workers=8)

    dict_dataset['test_loader'] = DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             generator=generator,
                                             num_workers=8)
    # nb_channels = dataMN.data.shape[1] * dataMN.data.shape[2]

    # limited_samples = 100
    # time_length = 300
    # data =dataMN.data[:limited_samples*2].flatten(start_dim=1,end_dim=2).unsqueeze(1).repeat(1, time_length, 1)
    # data = data*1e-3 + torch.randint_like(data,high=10)*2e-2
    # labels = dataMN.targets[:limited_samples*2]
    # print(data.shape)
    # print(labels.shape)
    # xtrain,xtest,ytrain,ytest = train_test_split(data,labels, test_size=0.2,stratify=labels,random_state=args.seed)
    # if args.nni:
    #     xtrain,xtest,ytrain,ytest = train_test_split(xtrain,ytrain, test_size=0.2,stratify=ytrain,random_state=args.seed)

    # ds_train = TensorDataset(xtrain, ytrain)
    # ds_test = TensorDataset(xtest, ytest)

    # Network parameters
    nb_input_copies = args.expansion
    nb_inputs = test_dataset.n_inputs * nb_input_copies
    nb_hidden = args.nb_hidden
    nb_outputs = test_dataset.n_classes

    print(f"nb_input_copies {nb_input_copies}")
    print(f"nb_inputs {nb_inputs}")
    print(f"nb_hidden {nb_hidden}")
    print(f"nb_outputs {nb_outputs}")

    dl_train = dict_dataset["train_loader"]
    dl_test = dict_dataset["test_loader"]
    # Learning parameters
    nb_epochs = args.nb_epochs

    # Neuron parameters
    tau_mem = args.tau_mem  # ms
    tau_syn = tau_mem / args.tau_ratio
    alpha = float(np.exp(-dt / tau_syn))
    beta = float(np.exp(-dt / tau_mem))

    fwd_weight_scale = args.fwd_weight_scale
    rec_weight_scale = args.weight_scale_factor * fwd_weight_scale

    ###########################################
    ##                Network                ##
    ###########################################


    if args.ALIF == True:
        dict_param = ALIF_dict_param
    else:
        dict_param = MN_dict_param
    C = 1
    print(dict_param)

    if args.shared_params:
        for param in dict_param:
            dict_param[param]["param"] = nn.Parameter(
                torch.Tensor([dict_param[param]["ini"]]),
                requires_grad=dict_param[param]["train"],
            )
    else:
        for param in dict_param:
            dict_param[param]["param"] = nn.Parameter(
                torch.Tensor(nb_inputs), requires_grad=dict_param[param]["train"]
            )
            dict_param[param]["param"].data.uniform_(
                dict_param[param]["ini"] * 0.9, dict_param[param]["ini"] * 1.1
            )
    if args.path_to_optimal_model is not None:
        # Load MN params from file:
        with open(Path(args.path_to_optimal_model).joinpath('Braille.json'), 'r') as f:
            loaded_data = json.load(f)
        for param in dict_param:
            dict_param[param]["param"] = nn.Parameter(
                torch.Tensor([loaded_data[param]]),
                requires_grad=False,
            )
        # # Load MN hyperparams:
        # with open(Path(args.path_to_optimal_model).joinpath('Braille_hyperparams.json'), 'r') as f:
        #     data = json.load(f)

    for param in dict_param:
        dict_param[param]["param"].to(device)
    # torch.autograd.set_detect_anomaly(True)
    if args.ALIF == True:
        l0 = ALIF_neuron(
                 nb_inputs=nb_inputs,
                 beta = dict_param["beta_alif"]["param"],
                 is_recurrent=False,
                 b_0=dict_param["b_0"]["param"],
                 dt=dt,
                 tau_adp=dict_param["tau_adp"]["param"],
                 beta_adapt=dict_param["beta_adapt"]["param"],
                 analog_input=True,
                 device = device)
    else:
        l0 = MN_neuron_sp(
                nb_inputs,
                firing_mode_dict[args.firing_mode],
                dt=dt,
                train=args.train,
                a=dict_param["a"]["param"],
                A1=dict_param["A1"]["param"],
                A2=dict_param["A2"]["param"],
                b=dict_param["b"]["param"],
                G=dict_param["G"]["param"],
                k1=dict_param["k1"]["param"],
                k2=dict_param["k2"]["param"],
                R1=dict_param["R1"]["param"],
                R2=dict_param["R2"]["param"],
                C=C,
            )
    network = nn.Sequential(
        Encoder(nb_inputs, args.norm, bias=0.0, nb_input_copies=nb_input_copies),
        l0,
        LIF_neuron(
            nb_inputs,
            nb_hidden,
            alpha,
            beta,
            is_recurrent=True,
            fwd_weight_scale=fwd_weight_scale,
            rec_weight_scale=rec_weight_scale,
        ),
        LIF_neuron(
            nb_hidden,
            nb_outputs,
            alpha,
            beta,
            is_recurrent=False,
            fwd_weight_scale=fwd_weight_scale,
            rec_weight_scale=rec_weight_scale,
        ),
    ).to(device)
    print(network)


    ###########################################
    ##               Training                ##
    ###########################################
    batch_size = args.batch_size
    if args.path_to_optimal_model is not None:
        print(' *** Recording activity post training ***')
        output_folder = Path('MN_output')
        output_folder.mkdir(parents=True, exist_ok=True)

        dl = dict_dataset
        for subset in dl.keys():
            folder = output_folder.joinpath('MNIST', subset)
            folder.mkdir(parents=True, exist_ok=True)
            for batch_idx, (x_local, y_local) in enumerate(dl[subset]):
                # Reset all the layers in the network
                for layer in network:
                    if hasattr(layer.__class__, "reset"):
                        layer.reset()

                l0_spk = []

                for t in range(x_local.shape[1]):
                    _ = network(x_local[:, t])
                    # Get the spikes and voltages from the MN neuron encoder
                    l0_spk.append(network[1].state.spk)

                l0_spk = torch.stack(l0_spk, dim=1)

                torch.save(l0_spk, folder.joinpath(f'GR_MNIST_b{batch_idx}_out.pt'))
                torch.save(y_local, folder.joinpath(f'GR_MNIST_b{batch_idx}_label.pt'))

    else:
        print(' *** Training model ***')
        ## Add the parameters from the LIF layers (2 and 3)
        my_list = ["2.", "3."]
        weight_params = [
            kv[1]
            for kv in filter(
                lambda kv: any([ele for ele in my_list if (ele in kv[0])]),
                network.named_parameters(),
            )
        ]
        param_list = [{"params": weight_params}]
        ## Add parameters form dict_param
        for param in dict_param:
            custom_param = [
                kv[1]
                for kv in filter(
                    lambda kv: any([ele for ele in [param] if (ele in kv[0])]),
                    network.named_parameters(),
                )
            ]
            if dict_param[param]["custom_lr"] is not None:
                param_list.append(
                    {"params": custom_param, "lr": dict_param[param]["custom_lr"]}
                )
            else:
                param_list.append({"params": custom_param})


        ## Create optimizer
        optimizer = torch.optim.Adamax(param_list, lr=args.lr, betas=(0.9, 0.995))
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=75,  # Number of iterations for the first restart
        #     T_mult=1,  # A factor increases TiTi​ after a restart
        #     eta_min=0,
        # )  # Minimum learning rate
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

        ttc_hist = []
        loss_hist = []
        accs_hist = [[], []]

        if args.log:
            #writer = SummaryWriter(comment="MN_WITH_GR_L1_MNIST")  # For logging purpose
            if args.nni:
                log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
                writer = SummaryWriter(log_dir=log_dir, comment="GR_MNIST")
            else:
                writer = SummaryWriter(comment="GR_MNIST")


        pbar = trange(nb_epochs,desc='Simulating')
        batches = trange(len(dl_train),desc='Training',leave=False)
        time = trange(1,desc='Time',leave=False)
        for e in pbar:
            local_loss = []
            local_loss_GR = []
            local_spk_count = []
            accs = []
            grad_dict_coll = []# accs: mean training accuracies for each batch
            for batch_idx, (x_local, y_local) in enumerate(dl_train):
                y_local = y_local[:,0]
                loss,acc,loss_GR,grad_dict,spk_count,recorder = training(x_local,y_local,device,network,log_softmax_fn,loss_fn,optimizer,args,dict_param,time)
                local_loss.append(loss)
                accs.append(acc)
                local_loss_GR.append(loss_GR)
                local_spk_count.append(spk_count.detach().cpu().numpy())
                grad_dict_coll.append(grad_dict)
                batches.update()
                # if batch_idx > 3:
                #     break

                if np.logical_or.reduce([torch.isnan(grad_dict[param]).cpu().numpy() for param in grad_dict if grad_dict[param] is not None]):
                    with open('filename.pickle', 'wb') as handle:
                        pickle.dump(recorder, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # scheduler.step()
            mean_loss = np.mean(local_loss)
            mean_loss_gr = np.mean(local_loss_GR)
            for key in grad_dict:
                if grad_dict[key] is not None:
                    grad_dict[key] = np.mean([grad_dict_coll[i][key].detach().cpu().numpy() for i in range(len(grad_dict_coll))])
            mean_spk_count = np.mean(np.concatenate(local_spk_count,axis=0))
            # mean_spk_count = -10
            loss_hist.append(mean_loss)
            # mean_accs: mean training accuracy of current epoch (average over all batches)
            mean_accs = np.mean(accs)
            accs_hist[0].append(mean_accs)
            with torch.no_grad():
                # Calculate test accuracy in each epoch on the testing dataset
                (
                    test_acc,
                    test_ttc,
                    l0_spk,
                    lif1_spk,
                    lif2_spk,
                    l0_mem,
                    lif1_mem,
                    lif2_mem,
                ) = compute_classification_accuracy(dl_test, network, True, device,args,args.fast,batches,time)
                accs_hist[1].append(test_acc)  # only safe best test
                ttc_hist.append(test_ttc)

                if args.log:
                    ###########################################
                    ##               Plotting                ##
                    ###########################################

                    # fig1 = plot_spikes(mn_spk.cpu())
                    # fig2 = plot_spikes(lif1_spk.cpu())
                    # fig3 = plot_spikes(lif2_spk.cpu())
                    #
                    # fig4 = plot_voltages(mn_mem.cpu())
                    # fig5 = plot_voltages(lif1_mem.cpu())
                    # fig6 = plot_voltages(lif2_mem.cpu())

                    ###########################################
                    ##                Logging                ##
                    ###########################################
                    if args.nni:
                        nni.report_intermediate_result(test_acc)

                    writer.add_scalar("Accuracy/test", test_acc, global_step=e)
                    writer.add_scalar("Accuracy/train", mean_accs, global_step=e)
                    # writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step=e)
                    # for idx, lr in enumerate(scheduler.get_last_lr()):
                    #     writer.add_scalar(f"lr{idx}", lr, global_step=e)
                    # writer.add_scalar("a", a, global_step=e)
                    writer.add_scalar("Loss/Local", mean_loss, global_step=e)
                    writer.add_scalar("Loss/GR", mean_loss_gr, global_step=e)
                    writer.add_scalar("spk count l2", mean_spk_count, global_step=e)
                    if args.shared_params:
                        for param in dict_param:
                            writer.add_scalar(
                                param, dict_param[param]["param"], global_step=e
                            )
                        for param in grad_dict:
                            if grad_dict[param] is not None:
                                writer.add_scalar(param+"_grad",grad_dict[param],global_step=e)
                    else:
                        for param in dict_param:
                            writer.add_histogram(
                                param, dict_param[param]["param"], global_step=e
                            )

                    # writer.add_histogram("w1", network[-2].weight, global_step=e)
                    # writer.add_histogram("w1_rec", network[-2].weight_rec, global_step=e)
                    # writer.add_histogram("w2", network[-1].weight, global_step=e)

            pbar.set_postfix_str(
                "Train accuracy: "
                + str(np.round(accs_hist[0][-1] * 100, 2))
                + "%. Test accuracy: "
                + str(np.round(accs_hist[1][-1] * 100, 2))
                + "%, Loss: "
                + str(np.round(mean_loss, 2))
            )

        if args.log:
            nni.report_final_result(test_acc)
            args_dict = args.__dict__
            args_dict.pop("log")
            args_dict.pop("data_path")
            for param in dict_param:
                for element in dict_param[param]:
                    if (element in ["ini", "train", "custom_lr"]) & (
                        dict_param[param][element] != None
                    ):
                        args_dict[param + "_" + element] = dict_param[param][element]
            writer.add_hparams(
                args_dict,
                {
                    "hparam/Accuracy/test": np.max(accs_hist[1]),
                    "hparam/Accuracy/train": np.max(accs_hist[0]),
                    "hparam/loss": np.min(loss_hist),
                },
                run_name=".",
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Encoding")
    parser.add_argument("--seed", type=int, default=6, help="Random seed. Default: 6")
    parser.add_argument(
        "--firing-mode",
        type=str,
        default="FA",
        choices=["FA", "SA", "MIX"],
        help="Choose between different firing modes",
    )
    parser.add_argument(
        "--norm",
        type=float,
        default=parameters_thenc["scale"],
        help="Data normalization",
    )
    parser.add_argument(
        "--upsample", type=float, default=1.0, help="Data upsample (default 100Hz)"
    )
    parser.add_argument(
        "--expansion",
        type=int,
        default=parameters_thenc["nb_input_copies"],
        help="Number of channel expansion (default: 1 (no expansion)).",
    )
    parser.add_argument(
        "--tau_mem",
        type=float,
        default=parameters_thenc["tau_mem"],
        help="Membrane time constant.",
    )
    parser.add_argument(
        "--tau_ratio",
        type=float,
        default=parameters_thenc["tau_ratio"],
        help="Tau ratio.",
    )
    parser.add_argument(
        "--fwd_weight_scale",
        type=float,
        default=parameters_thenc["fwd_weight_scale"],
        help="fwd_weight_scale.",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/data_braille_letters_all.pkl",
        help="The path where the " "dataset can be found",
    )

    parser.add_argument(
        "--weight_scale_factor",
        type=float,
        default=parameters_thenc["weight_scale_factor"],
        help="weight_scale_factor",
    )
    parser.add_argument(
        "--reg_spikes_l1",
        type=float,
        default=parameters_thenc["reg_spikes"],
        help="reg_spikes l1",
    )
    parser.add_argument(
        "--reg_neurons_l1",
        type=float,
        default=parameters_thenc["reg_neurons"],
        help="reg_neurons l1",
    )
    parser.add_argument(
        "--reg_spikes_l2",
        type=float,
        default=parameters_thenc["reg_spikes"],
        help="reg_spikes l2",
    )

    parser.add_argument(
        "--reg_neurons_l2",
        type=float,
        default=parameters_thenc["reg_neurons"],
        help="reg_neurons l2",
    )

    parser.add_argument(
        "--nb_epochs",
        type=int,
        default=parameters_thenc["nb_epochs"],
        help="number of epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=parameters_thenc["batch_size"],
        help="batch_size",
    )

    parser.add_argument(
        "--nb_hidden",
        type=int,
        default=parameters_thenc["nb_hidden"],
        help="number of hidden neurons",
    )

    parser.add_argument(
        "--shared_params",
        action="store_true",
        help="Train a single shared params set between neurons",
    )

    parser.add_argument(
        "--gr",
        type=float,
        default=0.2006232276576734,
        help="Gradient regularization",
    )
    parser.add_argument(
        "--ALIF",
        action="store_true",
        help="Use ALIF neurons instead of MN",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU",
    )
    parser.add_argument(
        "--nni",
        action="store_true",
        help="run with nni",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip saving mems",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning Rate",
    )
    parser.add_argument(
        "--path_to_optimal_model",
        type=str,
        default=None,  # None, #"./MN_params",
        help="path to folder that stores the parameters after training with nni (both MN params and hyperparams)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.02,  # None, #"./MN_params",
        help="Scaling dataset to neuron",
    )


    parser.add_argument("--log", action="store_true", help="Log on tensorboard.")

    parser.add_argument("--train", action="store_true", help="Train the MN neuron.")
    args = parser.parse_args()
    assert args.expansion > 0, "Expansion number should be greater that 0"

    if args.nni:
        PARAMS = nni.get_next_parameter()
        print(PARAMS)
        # Replace default args with new set
        d = vars(args)  # copy by reference (checked below)
        for key, val in PARAMS.items():
            d[key] = val
            assert (args.__dict__[key] == d[key])

    main(args)