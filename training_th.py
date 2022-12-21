import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
#import torchviz
import matplotlib.pyplot as plt

from tqdm import trange

import numpy as np

from datasets import load_and_extract_events
from parameters.MN_params import MNparams_dict, INIT_MODE
from models import Encoder, LIF_neuron, MN_neuron_sp
from auxiliary import compute_classification_accuracy_nomn, plot_spikes, plot_voltages

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

firing_mode_dict = {
    "FA": {"a": 5, "A1": 0, "A2": 0},
    "SA": {"a": 0, "A1": 0, "A2": 0},
    "MIX": {"a": 5, "A1": 5, "A2": -0.3},
}


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    ###########################################
    ##                Dataset                ##
    ###########################################
    upsample_fac = 1
    dt = (1 / 100.0) / upsample_fac
    file_freq = 100  # 40
    threshold = 2
    file_type = 'data_braille_letters_th_'
    # file_type = 'data_braille_letters_events_augmented_th'
    file_thr = str(threshold)
    file_name = "data/" + file_type + file_thr + '.pkl'  # '_rpNull'
    # file_name = "data/data_braille_letters_th_2.pkl"

    param_filename = 'parameters_th' + str(threshold) + '.txt'
    file_name_parameters = "parameters/" + param_filename
    params = {}
    with open(file_name_parameters) as file:
        for line in file:
            (key, value) = line.split()
            if key == 'time_bin_size' or key == 'nb_input_copies':
                params[key] = int(value)
            else:
                params[key] = np.double(value)
    ds_train, _, ds_test, labels, nb_channels, data_steps = load_and_extract_events(
        params, file_name)
    # data, labels, _, _, _, _ = load_data(file_name, upsample_fac)
    # nb_channels = data.shape[-1]
    #
    # x_train, x_test, y_train, y_test = train_test_split(
    #     data, labels, test_size=0.2, shuffle=True, stratify=labels
    # )
    #
    # ds_train = TensorDataset(x_train, y_train)
    # ds_test = TensorDataset(x_test, y_test)

    # Network parameters
    nb_input_copies = args.expansion
    nb_inputs = nb_channels * nb_input_copies
    nb_hidden = 450
    nb_outputs = len(np.unique(labels))

    # Learning parameters
    nb_epochs = 300

    # Neuron parameters
    tau_mem = args.tau_mem  # ms
    tau_syn = tau_mem / args.tau_ratio
    alpha = float(np.exp(-dt / tau_syn))
    beta = float(np.exp(-dt / tau_mem))

    encoder_weight_scale = 1.0
    fwd_weight_scale = args.fwd_weight_scale
    rec_weight_scale = args.weight_scale_factor * fwd_weight_scale

    ###########################################
    ##                Network                ##
    ###########################################


    torch.autograd.set_detect_anomaly(True)
    network = nn.Sequential(
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
    batch_size = 128

    optimizer = torch.optim.Adamax(network.parameters(), lr=0.005, betas=(0.9, 0.995))
    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

    ttc_hist = []
    loss_hist = []
    accs_hist = [[], []]

    if args.log:
        writer = SummaryWriter(comment="training_th")  # For logging purpose

    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    pbar = trange(nb_epochs)
    for e in pbar:
        local_loss = []
        accs = []  # accs: mean training accuracies for each batch
        for x_local, y_local in dl_train:
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

            mn_spk = []
            lif1_spk = []
            lif2_spk = []

            mn_mem = []
            lif1_mem = []
            lif2_mem = []

            for t in range(x_local.shape[1]):
                out = network(x_local[:, t])

                # Get the spikes and voltages from the first LIF
                lif1_spk.append(network[0].state.S)
                lif1_mem.append(network[0].state.mem)

                # Get the spikes and voltages from the second LIF
                lif2_spk.append(network[1].state.S)
                lif2_mem.append(network[1].state.mem)

            # mn_spk = torch.stack(mn_spk, dim=1)
            # mn_mem = torch.stack(mn_mem, dim=1)
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
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())

            with torch.no_grad():
                # compare to labels
                _, am = torch.max(m, 1)  # argmax over output units
                tmp = np.mean((y_local == am).detach().cpu().numpy())
                accs.append(tmp)

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
                lif1_spk,
                lif2_spk,
                lif1_mem,
                lif2_mem,
            ) = compute_classification_accuracy_nomn(dl_test, network, True, device)
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

                writer.add_scalar("Accuracy/test", test_acc, global_step=e)
                writer.add_scalar("Accuracy/train", mean_accs, global_step=e)
                #writer.add_scalar("a", a, global_step=e)
                writer.add_scalar("Loss", mean_loss, global_step=e)


                # writer.add_figure("MN spikes", fig1, global_step=e)
                # writer.add_figure("LIF1 spikes", fig2, global_step=e)
                # writer.add_figure("LIF2 spikes", fig3, global_step=e)
                # writer.add_figure("MN voltage", fig4, global_step=e)
                # writer.add_figure("LIF1 voltage", fig5, global_step=e)
                # writer.add_figure("LIF2 voltage", fig6, global_step=e)
                writer.add_histogram("w1", network[-2].weight, global_step=e)
                writer.add_histogram("w1_rec", network[-2].weight_rec, global_step=e)
                writer.add_histogram("w2", network[-1].weight, global_step=e)

        pbar.set_postfix_str(
            "Train accuracy: "
            + str(np.round(accs_hist[0][-1] * 100, 2))
            + "%. Test accuracy: "
            + str(np.round(accs_hist[1][-1] * 100, 2))
            + "%, Loss: "
            + str(np.round(mean_loss, 2))
        )

    if args.log:
        args_dict = args.__dict__
        args_dict.pop("log")
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
    parser.add_argument("--norm", type=float, default=10.0, help="Data normalization")
    parser.add_argument(
        "--upsample", type=float, default=1.0, help="Data upsample (default 100Hz)"
    )
    parser.add_argument(
        "--expansion",
        type=int,
        default=1,
        help="Number of channel expansion (default: 1 (no expansion)).",
    )
    parser.add_argument(
        "--tau_mem", type=float, default=0.02, help="Membrane time constant."
    )
    parser.add_argument("--tau_ratio", type=float, default=2, help="Tau ratio.")
    parser.add_argument(
        "--fwd_weight_scale", type=float, default=1, help="fwd_weight_scale."
    )
    parser.add_argument(
        "--weight_scale_factor", type=float, default=0.01, help="weight_scale_factor"
    )
    parser.add_argument("--reg_spikes", type=float, default=0.004, help="reg_spikes")
    parser.add_argument(
        "--reg_neurons", type=float, default=0.000001, help="reg_neurons"
    )
    parser.add_argument("--shared_params",action="store_true", help="Train a single shared params set between neurons")

    parser.add_argument("--log", action="store_true", help="Log on tensorboard.")

    parser.add_argument("--train", action="store_true", help="Train the MN neuron.")
    args = parser.parse_args()
    assert args.expansion > 0, "Expansion number should be greater that 0"
    main(args)
