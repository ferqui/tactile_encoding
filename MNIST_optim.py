import json
import os
import random
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from dataset_optim import MNISTDataset
from models import ALIF_neuron, Encoder, LIF_neuron, MN_neuron_sp

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


class Telegram_bot():
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id

    def send_message(self, message):
        requests.get(f'https://api.telegram.org/bot{self.token}/sendMessage?chat_id={self.chat_id}&text={message}')


@torch.no_grad()
def compute_classification_accuracy(dataset, network, early, device, args, fast=True):
    accs = []
    x_test, y_test = dataset
    for x_local, y_local in tqdm(zip(x_test, y_test)):
        for layer in network:
            if hasattr(layer.__class__, "reset"):
                layer.reset()

        lif2_spk = []
        for t in range(x_local.shape[1]):
            out = network(x_local[:, t] * args.gain)
            # Get the spikes and voltages from the second LIF
            lif2_spk.append(out)
        lif2_spk = torch.stack(lif2_spk, dim=1)
        lif2_sum = torch.sum(lif2_spk, dim=1)

        # with output spikes
        _, am = torch.max(lif2_sum, 1)  # argmax over output units
        # compare to labels
        tmp = torch.mean((y_local == am).float()).item()
        accs.append(tmp)

    return np.mean(accs)


def main(args):
    device = (
        torch.device("cuda:0")
        if (torch.cuda.is_available() & args.gpu)
        else torch.device("cpu")
    )
    print(device)

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(0)
    if args.path_to_optimal_model is not None:
        if ',' in args.path_to_optimal_model:
            args.path_to_optimal_model = args.path_to_optimal_model.split(',')
    if args.telegram_bot_token_path is not None:
        with open(args.telegram_bot_token_path) as f:
            token = f.read()
        bot = Telegram_bot(token, args.telegram_bot_chat_id)

    ###########################################
    ##                Dataset                ##
    ###########################################
    upsample_fac = 1
    dt = (1 / 100.0) / upsample_fac
    dataset = MNISTDataset(args.num_train, args.num_test, args.val_size, args.batch_size, 3, dt_sec=1e-2, v_max=0.2,
                           add_noise=True, gain=1.0, compressed=args.compressed, encoder_model=args.encoder_model)

    # Network parameters
    nb_input_copies = args.expansion
    nb_inputs = dataset.n_inputs * nb_input_copies
    nb_hidden = args.nb_hidden
    nb_outputs = dataset.n_classes
    nb_epochs = args.nb_epochs

    print(f"nb_input_copies {nb_input_copies}")
    print(f"nb_inputs {nb_inputs}")
    print(f"nb_hidden {nb_hidden}")
    print(f"nb_outputs {nb_outputs}")

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
        if type(args.path_to_optimal_model) is list:
            optimal_model = args.path_to_optimal_model[args.seed]
        else:
            optimal_model = args.path_to_optimal_model
        with open(Path(optimal_model), 'r') as f:
            loaded_data = json.load(f)
        for param in dict_param:
            dict_param[param]["param"] = nn.Parameter(
                torch.Tensor([loaded_data[param]]),
                requires_grad=False,
            )
    for param in dict_param:
        dict_param[param]["param"].to(device)
    if args.ALIF == True:
        l0 = ALIF_neuron(
            nb_inputs=nb_inputs,
            beta=dict_param["beta_alif"]["param"],
            is_recurrent=False,
            b_0=dict_param["b_0"]["param"],
            dt=dt,
            tau_adp=dict_param["tau_adp"]["param"],
            beta_adapt=dict_param["beta_adapt"]["param"],
            analog_input=True,
            device=device)
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
    if args.path_to_optimal_model is not None:
        print(' *** Recording activity post training ***')
        if type(args.path_to_optimal_model) is list:
            model = args.path_to_optimal_model[args.seed].split('/')[-1].split('.')[0]
        else:
            model = args.path_to_optimal_model.split('/')[-1].split('.')[0]
        output_folder = Path(args.new_dataset_output_folder).joinpath(model)
        output_folder.mkdir(parents=True, exist_ok=True)
        dl = {'train': dataset.get_train(device), 'test': dataset.get_test(device)}
        for subset in dl.keys():
            folder = output_folder.joinpath(subset)
            folder.mkdir(parents=True, exist_ok=True)
            for batch_idx, (x_local, y_local) in enumerate(zip(dl[subset][0], dl[subset][1])):
                # Reset all the layers in the network
                for layer in network:
                    if hasattr(layer.__class__, "reset"):
                        layer.reset()
                l0_spk = []
                for t in range(x_local.shape[1]):
                    _ = network(x_local[:, t] * args.gain)
                    # Get the spikes and voltages from the MN neuron encoder
                    l0_spk.append(network[1].state.spk)
                l0_spk = torch.stack(l0_spk, dim=1)
                # l0_spk = l0_spk.to(bool).to_sparse()
                l0_spk = np.array(l0_spk.cpu().to(bool))
                l0_spk = np.packbits(l0_spk, axis=0)
                np.save(folder.joinpath(f'{model}_{batch_idx}.npy'), l0_spk, allow_pickle=True)
                # torch.save(l0_spk, folder.joinpath(f'{model}_{batch_idx}.pt'))
                torch.save(y_local, folder.joinpath(f'{model}_{batch_idx}_label.pt'))
        raise ValueError('Done recording activity')
    ###########################################
    ##               Training                ##
    ###########################################
    print(' *** Training model ***')
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True, check_nan=True)

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
    gr_reg_params = []
    ## Add parameters form dict_param
    for param in dict_param:
        if dict_param[param]["train"]:
            if dict_param[param]["custom_lr"] is not None:
                param_list.append(
                    {"params": dict_param[param]["param"], "lr": dict_param[param]["custom_lr"]}
                )
            else:
                param_list.append({"params": dict_param[param]["param"]})

            gr_reg_params.append(dict_param[param]["param"])

    ## Create optimizer
    optimizer = torch.optim.Adamax(param_list, lr=args.lr, betas=(0.9, 0.995))
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()

    loss_hist = []
    accs_hist = [[], []]

    if args.log:
        writer = SummaryWriter(comment="GR_MNIST_optim_" + "seed-" + str(args.seed))

    pbar = trange(nb_epochs, desc='Simulating')
    for e in pbar:
        x_train, y_train = dataset.get_train(device)
        ## Training
        accs = []
        total_loss = []
        mean_firing = [[], []]
        network.train()
        for x_local, y_local in tqdm(zip(x_train, y_train)):
            ## Reset network internal states
            for layer in network:
                if hasattr(layer.__class__, "reset"):
                    layer.reset()

            lif1_spk = torch.zeros(args.batch_size, nb_hidden, device=device)
            lif2_spk = torch.zeros(args.batch_size, nb_outputs, device=device)
            for t in range(x_local.shape[1]):
                out = network(x_local[:, t] * args.gain)
                lif1_spk = lif1_spk + network[2].state.S
                lif2_spk = lif2_spk + network[3].state.S
            log_p_y = log_softmax_fn(lif2_spk)

            # Here we can set up our regularizer loss
            reg_loss = args.reg_spikes_l1 * torch.mean(
                lif1_spk)  # e.g., L1 loss on total number of spikes (original: 1e-3)
            reg_loss += args.reg_neurons_l1 * torch.mean(
                torch.sum(lif1_spk, dim=0) ** 2
            )  # L1 loss on spikes per neuron (original: 2e-6)
            reg_loss += args.reg_spikes_l2 * torch.mean(lif2_spk)  # L2 loss on output layer spikes (original: 1e-3)
            reg_loss += args.reg_neurons_l2 * torch.mean(
                torch.sum(lif2_spk, dim=0) ** 2)  # L2 loss on output layer spikes per neuron (original: 2e-6)
            ### regularizer for silent neurons
            reg_loss += args.reg_silent_neurons_gain * nn.ReLU()(
                torch.mean(args.reg_silent_neurons_th - lif2_spk)) ** 2  # penalize silent neurons (original: 1e-5)

            # Calculate loss
            loss_val = loss_fn(log_p_y, y_local.long()) + reg_loss
            optimizer.zero_grad()

            GR_loss = 0
            if args.gr > 0:
                grads = torch.autograd.grad(loss_val, gr_reg_params, create_graph=True)
                for grad in grads:
                    GR_loss += torch.abs(grad).sum()

            loss_DB = loss_val + args.gr * GR_loss
            loss_DB.backward()

            grad_dict = {}
            for param in dict_param:
                if dict_param[param]["param"].grad is not None:
                    grad_dict[param] = dict_param[param]["param"].grad
            optimizer.step()

            with torch.no_grad():
                # compare to labels
                _, am = torch.max(lif2_spk, 1)  # argmax over output units
                accuracy = torch.mean((y_local == am).float()).item()
                accs.append(accuracy)
                total_loss.append(loss_DB.item())
                mean_firing[0].append(torch.mean(lif1_spk, dim=1).mean().item())
                mean_firing[1].append(torch.mean(lif2_spk, dim=1).mean().item())

        train_acc = np.mean(accs)
        mean_loss = np.mean(total_loss)
        mean_firing[0] = np.mean(mean_firing[0])
        mean_firing[1] = np.mean(mean_firing[1])

        loss_hist.append(mean_loss)
        accs_hist[0].append(train_acc)

        ## Evaluation step
        network.eval()
        test_acc = compute_classification_accuracy(dataset.get_val(), network, True, device, args, args.fast)
        accs_hist[1].append(test_acc)

        if args.log:
            writer.add_scalar("Accuracy/test", test_acc, global_step=e)
            writer.add_scalar("Accuracy/train", train_acc, global_step=e)
            writer.add_scalar("Loss", mean_loss, global_step=e)
            writer.add_scalar("Firing rate/LIF1", mean_firing[0], global_step=e)
            writer.add_scalar("Firing rate/LIF2", mean_firing[1], global_step=e)
            writer.add_histogram("LIF1", network[2].weight.grad, global_step=e)
            writer.add_histogram("LIF2", network[3].weight.grad, global_step=e)
            if args.shared_params:
                for param in dict_param:
                    writer.add_scalar(
                        param, dict_param[param]["param"], global_step=e
                    )
                for param in grad_dict:
                    writer.add_scalar(param + "_grad", grad_dict[param], global_step=e)
            else:
                for param in dict_param:
                    writer.add_histogram(
                        param, dict_param[param]["param"], global_step=e
                    )

        pbar.set_postfix_str(
            "Train accuracy: "
            + str(np.round(train_acc * 100, 2))
            + "%. Test accuracy: "
            + str(np.round(test_acc * 100, 2))
            + "%, Loss: "
            + str(np.round(mean_loss, 2))
        )
        if args.telegram_bot_token_path is not None:
            bot.send_message(
                "Epoch"
                + str(e)
                + "Train accuracy: "
                + str(np.round(train_acc * 100, 2))
                + "%. Test accuracy: "
                + str(np.round(test_acc * 100, 2))
                + "%, Loss: "
                + str(np.round(mean_loss, 2))
            )

    if args.log:
        # nni.report_final_result(test_acc)
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

    parameters_thenc = {}
    with open("parameters/parameters_thenc.txt") as f:
        for line in f:
            (key, val) = line.split()
            parameters_thenc[key] = val

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
        "--num-train",
        type=int,
        default=6480,
        help="Number of channel expansion (default: 1 (no expansion)).",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=1620,
        help="Number of channel expansion (default: 1 (no expansion)).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
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
        "--reg_silent_neurons_gain",
        type=float,
        default=5,
        help="reg for avoiding silent neurons (gain)",
    )
    parser.add_argument(
        "--reg_silent_neurons_th",
        type=float,
        default=20,
        help="reg for avoiding silent neurons (threshold)",
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
        "--gain",
        type=float,
        default=0.02,  # None, #"./MN_params",
        help="Scaling dataset to neuron",
    )
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Use reduced dataset",
    )
    parser.add_argument("--detect_anomaly", action="store_true", help="Detect anomaly.")
    parser.add_argument('--compressed', action='store_true',
                        help='Use dataset compressed through an autoencoder with 24 channels')
    parser.add_argument("--encoder_model", type=str, default='./data/784MNIST_2_6MNIST.pt',
                        help="Path to encoder model.")

    parser.add_argument("--log", action="store_true", help="Log on tensorboard.")

    parser.add_argument("--train", action="store_true", help="Train the MN neuron.")
    parser.add_argument("--telegram_bot_token_path", type=str, default=None, help="Path to telegram bot token.")
    parser.add_argument("--telegram_bot_chat_id", type=str, default='15905296', help="Chat id for telegram bot.")
    parser.add_argument("--path_to_optimal_model", type=str, default=None,
                        help="Path to optimal model (it only simulates the first layer using already trained model).")
    parser.add_argument("--new_dataset_output_folder", type=str, default='MN_output',
                        help="Path to folder where to save the new dataset.")

    args = parser.parse_args()
    assert args.expansion > 0, "Expansion number should be greater that 0"

    main(args)