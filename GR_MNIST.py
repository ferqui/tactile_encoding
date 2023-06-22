import nni
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# import torchviz
import matplotlib.pyplot as plt

from tqdm import trange

import numpy as np

from datasets import load_data

from parameters.MN_params import MNparams_dict, INIT_MODE
from models import Encoder, LIF_neuron, MN_neuron_sp, ALIF_neuron
from auxiliary import compute_classification_accuracy, plot_spikes, plot_voltages

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST

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


def main(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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
    dataMN = MNIST(
        root="data",
        train=True,
        download=True,
        )

    nb_channels = dataMN.data.shape[1] * dataMN.data.shape[2]

    limited_samples = 300
    train_data = dataMN.train_data[:limited_samples].flatten(start_dim=1,end_dim=2).unsqueeze(1).repeat(1, 300, 1)
    train_labels = dataMN.train_labels[:limited_samples]
    test_data = dataMN.test_data[:limited_samples].flatten(start_dim=1,end_dim=2).unsqueeze(1).repeat(1, 300, 1)
    test_labels = dataMN.test_labels[:limited_samples]
    train_data_noise = train_data + torch.randint_like(train_data,high=10)*0.05
    test_data_noise = test_data + torch.randint_like(test_data,high=10) * 0.05
    ds_train = TensorDataset(train_data_noise, train_labels)
    ds_test = TensorDataset(test_data_noise, test_labels)

    # Network parameters
    nb_input_copies = args.expansion
    nb_inputs = nb_channels * nb_input_copies
    nb_hidden = args.nb_hidden
    nb_outputs = len(np.unique(dataMN.train_labels))

    print(f"nb_input_copies {nb_input_copies}")
    print(f"nb_inputs {nb_inputs}")
    print(f"nb_hidden {nb_hidden}")
    print(f"nb_outputs {nb_outputs}")

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

    for param in dict_param:
        dict_param[param]["param"].to(device)

    ###########################################
    ##               Training                ##
    ###########################################
    batch_size = args.batch_size

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
    optimizer = torch.optim.Adamax(param_list, lr=0.005, betas=(0.9, 0.995))
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
        writer = SummaryWriter(comment="MN_WITH_GR_L1")  # For logging purpose
    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    pbar = trange(nb_epochs)
    for e in pbar:
        local_loss = []
        accs = []  # accs: mean training accuracies for each batch
        for batch_idx, (x_local, y_local) in enumerate(dl_train):
            pbar.set_description(f"{batch_idx}/{len(dl_train)}")
            x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(
                device, non_blocking=True
            )

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

            for t in range(x_local.shape[1]):
                out = network(x_local[:, t])

                # Get the spikes and voltages from the MN neuron encoder
                l0_spk.append(network[1].state.spk)
                l0_mem.append(network[1].state.V)

                # Get the spikes and voltages from the first LIF
                lif1_spk.append(network[2].state.S)
                lif1_mem.append(network[2].state.mem)

                # Get the spikes and voltages from the second LIF
                lif2_spk.append(network[3].state.S)
                lif2_mem.append(network[3].state.mem)

            l0_spk = torch.stack(l0_spk, dim=1)
            l0_events = np.where(l0_spk[0, :, :].cpu().detach().numpy())
            plt.scatter(l0_events[0], l0_events[1], s=0.1)
            plt.show()
            l0_mem = torch.stack(l0_mem, dim=1)
            lif1_spk = torch.stack(lif1_spk, dim=1)
            lif1_mem = torch.stack(lif1_mem, dim=1)
            lif2_spk = torch.stack(lif2_spk, dim=1)
            lif2_mem = torch.stack(lif2_mem, dim=1)
            m = torch.sum(lif2_spk, 1)  # sum over time
            log_p_y = log_softmax_fn(m)

            # Here we can set up our regularizer loss
            reg_loss = args.reg_spikes * torch.mean(
                torch.sum(lif1_spk, 1)
            )  # e.g., L1 loss on total number of spikes (original: 1e-3)
            reg_loss += args.reg_neurons * torch.mean(
                torch.sum(torch.sum(lif1_spk, dim=0), dim=0) ** 2
            )  # L2 loss on spikes per neuron (original: 2e-6)

            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local) + reg_loss

            optimizer.zero_grad()
            # loss_val.backward()
            loss_val.backward(create_graph=True)  # backpropagation of original loss
            loss_DB = args.gr * sum(
                [
                    torch.abs(kv[1]["param"].grad).sum()
                    for kv in filter(lambda kv: kv[1]["train"], dict_param.items())
                ]
            )  # computing GR term
            loss_DB.backward()  # backpropagation of GR ter
            optimizer.step()
            local_loss.append(loss_val.item())

            with torch.no_grad():
                # compare to labels
                _, am = torch.max(m, 1)  # argmax over output units
                tmp = np.mean((y_local == am).detach().cpu().numpy())
                accs.append(tmp)

        # scheduler.step()
        mean_loss = np.mean(local_loss)
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
            ) = compute_classification_accuracy(dl_test, network, True, device)
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
                nni.report_intermediate_result(test_acc)

                writer.add_scalar("Accuracy/test", test_acc, global_step=e)
                writer.add_scalar("Accuracy/train", mean_accs, global_step=e)
                # writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step=e)
                # for idx, lr in enumerate(scheduler.get_last_lr()):
                #     writer.add_scalar(f"lr{idx}", lr, global_step=e)
                # writer.add_scalar("a", a, global_step=e)
                writer.add_scalar("Loss", mean_loss, global_step=e)
                if args.shared_params:
                    for param in dict_param:
                        writer.add_scalar(
                            param, dict_param[param]["param"], global_step=e
                        )
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
    parser.add_argument("--seed", type=int, default=-1, help="Random seed. Default: -1")
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
        "--reg_spikes",
        type=float,
        default=parameters_thenc["reg_spikes"],
        help="reg_spikes",
    )
    parser.add_argument(
        "--reg_neurons",
        type=float,
        default=parameters_thenc["reg_neurons"],
        help="reg_neurons",
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
        default=1.0,
        help="Gradient regularization",
    )
    parser.add_argument(
        "--ALIF",
        action="store_true",
        help="Use ALIF neurons instead of MN",
    )

    parser.add_argument("--log", action="store_true", help="Log on tensorboard.")

    parser.add_argument("--train", action="store_true", help="Train the MN neuron.")
    args = parser.parse_args()
    assert args.expansion > 0, "Expansion number should be greater that 0"
    main(args)
