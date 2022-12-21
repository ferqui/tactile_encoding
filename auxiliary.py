import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_spikes(spikes, idx=0):
    t, idx = np.where(spikes[idx].cpu().numpy())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x=t, y=idx, s=0.5)

    return fig


def plot_voltages(voltage, idx=0):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(voltage[idx])

    return fig


@torch.no_grad()
def compute_classification_accuracy(dataset, network, early, device):
    accs = []
    multi_accs = []
    ttc = None
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(
            device, non_blocking=True
        )

        for layer in network:
            if hasattr(layer.__class__, "reset"):
                layer.reset()

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

        # with output spikes
        m = torch.sum(lif2_spk, 1)  # sum over time
        _, am = torch.max(m, 1)  # argmax over output units
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

@torch.no_grad()
def compute_classification_accuracy_nomn(dataset, network, early, device):
    accs = []
    multi_accs = []
    ttc = None
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(
            device, non_blocking=True
        )

        for layer in network:
            if hasattr(layer.__class__, "reset"):
                layer.reset()

        lif1_spk = []
        lif2_spk = []

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

        lif1_spk = torch.stack(lif1_spk, dim=1)
        lif1_mem = torch.stack(lif1_mem, dim=1)
        lif2_spk = torch.stack(lif2_spk, dim=1)
        lif2_mem = torch.stack(lif2_mem, dim=1)

        # with output spikes
        m = torch.sum(lif2_spk, 1)  # sum over time
        _, am = torch.max(m, 1)  # argmax over output units
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

    return np.mean(accs), ttc, lif1_spk, lif2_spk, lif1_mem, lif2_mem
