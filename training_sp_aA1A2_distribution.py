import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
#import torchviz
import matplotlib.pyplot as plt

from tqdm import trange

import numpy as np

from datasets import load_data
from parameters.MN_params import MNparams_dict, INIT_MODE
from models import Encoder, LIF_neuron, MN_neuron_sp
from auxiliary import compute_classification_accuracy, plot_spikes, plot_voltages

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
    file_name = "data/data_braille_letters_all.pkl"
    data, labels, _, _, _, _ = load_data(file_name, upsample_fac)
    nb_channels = data.shape[-1]

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)

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

    # a = torch.empty((nb_inputs,))
    # nn.init.normal_(
    #     a, mean=MNparams_dict[INIT_MODE][0], std=fwd_weight_scale / np.sqrt(nb_inputs))

    # A1 = torch.empty((nb_inputs,))
    # nn.init.normal_(
    #     A1, mean=MNparams_dict[INIT_MODE][1], std=fwd_weight_scale / np.sqrt(nb_inputs))

    # A2 = torch.empty((nb_inputs,))
    # nn.init.normal_(
    #     A2, mean=MNparams_dict[INIT_MODE][2], std=fwd_weight_scale / np.sqrt(nb_inputs))

    # a = -40
    a_ini = 5
    # A1 = 1.1
    A1_ini = 0
    # A2 = -1
    A2_ini = 0
    b = 10
    G = 50
    k1 = 200
    k2 = 20
    R1 = 0
    R2 = 1
    C = 1
    if args.shared_params:
        a = nn.Parameter(torch.Tensor([a_ini]), requires_grad=True)
    else:
        a = nn.Parameter(torch.Tensor(nb_inputs), requires_grad=True)
        a.data.uniform_(a_ini*0.9,a_ini*1.1)
    if args.shared_params:
        A1 = nn.Parameter(torch.Tensor([A1_ini]), requires_grad=True)
    else:
        A1 = nn.Parameter(torch.Tensor(nb_inputs), requires_grad=True)
        A1.data.uniform_(A1_ini*0.9,A1_ini*1.1)
    if args.shared_params:
        A2 = nn.Parameter(torch.Tensor([A2_ini]), requires_grad=True)
    else:
        A2 = nn.Parameter(torch.Tensor(nb_inputs), requires_grad=True)
        A2.data.uniform_(A2_ini * 0.9, A2_ini * 1.1)
    b = nn.Parameter(torch.Tensor([b]), requires_grad=False)
    G = nn.Parameter(torch.Tensor([G]), requires_grad=False)
    k1 = nn.Parameter(torch.Tensor([k1]), requires_grad=False)
    k2 = nn.Parameter(torch.Tensor([k2]), requires_grad=False)
    R1 = nn.Parameter(torch.Tensor([R1]), requires_grad=False)
    R2 = nn.Parameter(torch.Tensor([R2]), requires_grad=False)
    C = nn.Parameter(torch.Tensor([C]), requires_grad=False)
    torch.autograd.set_detect_anomaly(True)
    network = nn.Sequential(
        Encoder(nb_inputs, args.norm, bias=0.0, nb_input_copies=nb_input_copies),
        MN_neuron_sp(
            nb_inputs, firing_mode_dict[args.firing_mode], dt=dt, train=args.train, a = a, A1 = A1, A2 = A2,  b=b, G=G, k1=k1, k2=k2, R1=R1, R2=R2, C = C
        ),
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
    a.to(device)
    A1.to(device)
    A2.to(device)
    b.to(device)
    G.to(device)
    k1.to(device)
    k2.to(device)
    R1.to(device)
    R2.to(device)
    ###########################################
    ##               Training                ##
    ###########################################
    batch_size = 128
    my_list = ['2.', '3.']
    custom_lr = {'a': 1.1,'A1':0.1,'A2':0.011}
    weight_params = [kv[1] for kv in
                     filter(lambda kv: any([ele for ele in my_list if (ele in kv[0])]), network.named_parameters())]
    param_list = [{'params':weight_params}]
    for param in custom_lr:
        custom_param = [kv[1] for kv in
                         filter(lambda kv: any([ele for ele in [param] if (ele in kv[0])]),
                                network.named_parameters())]
        param_list.append({'params':custom_param,'lr' : custom_lr[param]})
    neuron_params = [kv[1] for kv in
                     filter(lambda kv: not any([ele for ele in my_list if (ele in kv[0])]), network.named_parameters())]
    optimizer = torch.optim.Adamax(param_list, lr=0.005, betas=(0.9, 0.995))
    # optimizer = torch.optim.Adamax(network.parameters(), lr=0.005, betas=(0.9, 0.995))
    optimizer = torch.optim.Adamax([{'params': weight_params}, {'params': neuron_params, 'lr': 0.05}], lr=0.005,
                                   betas=(0.9, 0.995))
    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

    ttc_hist = []
    loss_hist = []
    accs_hist = [[], []]

    if args.log:
        writer = SummaryWriter(comment="training_aA1A2_distribution")  # For logging purpose

    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    pbar = trange(nb_epochs)
    parameter_rec = {'a': [], 'A1':[], 'A2':[],'b':[], 'G':[], 'k1':[], 'k2':[],'R1':[], 'R2':[],'w1':[],'w1_rec':[],'w2':[]}
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

                # Get the spikes and voltages from the MN neuron encoder
                mn_spk.append(network[1].state.spk)
                mn_mem.append(network[1].state.V)

                # Get the spikes and voltages from the first LIF
                lif1_spk.append(network[2].state.S)
                lif1_mem.append(network[2].state.mem)

                # Get the spikes and voltages from the second LIF
                lif2_spk.append(network[3].state.S)
                lif2_mem.append(network[3].state.mem)

            mn_spk = torch.stack(mn_spk, dim=1)
            mn_mem = torch.stack(mn_mem, dim=1)
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
                mn_spk,
                lif1_spk,
                lif2_spk,
                mn_mem,
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

                writer.add_scalar("Accuracy/test", test_acc, global_step=e)
                writer.add_scalar("Accuracy/train", mean_accs, global_step=e)
                #writer.add_scalar("a", a, global_step=e)
                writer.add_scalar("Loss", mean_loss, global_step=e)
                if args.shared_params:
                    writer.add_scalar("a",a,global_step=e)
                    writer.add_scalar("A1", A1, global_step=e)
                    writer.add_scalar("A2", A2, global_step=e)

                else:
                    writer.add_histogram("a",a, global_step= e)
                    writer.add_histogram("A1",A1, global_step= e)
                    writer.add_histogram("A2",A2, global_step= e)


                # writer.add_figure("MN spikes", fig1, global_step=e)
                # writer.add_figure("LIF1 spikes", fig2, global_step=e)
                # writer.add_figure("LIF2 spikes", fig3, global_step=e)
                # writer.add_figure("MN voltage", fig4, global_step=e)
                # writer.add_figure("LIF1 voltage", fig5, global_step=e)
                # writer.add_figure("LIF2 voltage", fig6, global_step=e)
                writer.add_histogram("w1", network[-2].weight, global_step=e)
                writer.add_histogram("w1_rec", network[-2].weight_rec, global_step=e)
                writer.add_histogram("w2", network[-1].weight, global_step=e)
                writer.add_scalar("b", b, global_step=e)
                writer.add_scalar("G", G, global_step=e)
                writer.add_scalar("k1", k1, global_step=e)
                writer.add_scalar("k2", k2, global_step=e)
                writer.add_scalar("R1", R1, global_step=e)
                writer.add_scalar("R2", R2, global_step=e)

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

