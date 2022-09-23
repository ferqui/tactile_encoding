import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_spikes(spikes):
    _, t, idx = np.where(spikes.cpu().numpy())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x = t, y = idx, s=.5)

    return fig

@torch.no_grad()
def compute_classification_accuracy(params, dataset, network, early, device):
    accs = []
    multi_accs = []
    ttc = None
    for x_local, y_local in dataset:
        x_local, y_local = x_local.to(device, non_blocking=True), y_local.to(device, non_blocking=True)

        for layer in network:
            if hasattr(layer.__class__, 'reset'):
                layer.reset()

        spk_hidden = []
        s_out_rec = []
        for t in range(params['data_steps']):
            out = network(x_local[:,t])

            spk_hidden.append(network[-2].state.S)
            s_out_rec.append(out)
        spk_hidden = torch.stack(spk_hidden, dim=1)
        s_out_rec = torch.stack(s_out_rec, dim=1)

        # with output spikes
        m = torch.sum(s_out_rec, 1)  # sum over time
        _, am = torch.max(m, 1)  # argmax over output units
        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
        
        if early:
            accs_early = []
            for t in range(s_out_rec.shape[1] - 1):
                # with spiking output layer
                m_early = torch.sum(s_out_rec[:, :t + 1, :], 1)  # sum over time
                _, am_early = torch.max(m_early, 1)  # argmax over output units
                # compare to labels
                tmp_early = np.mean((y_local == am_early).detach().cpu().numpy())
                accs_early.append(tmp_early)
            multi_accs.append(accs_early)

    if early:
        max_time = int(54*25) #ms
        time_bin_size = int(params['time_bin_size']) # ms
        time = range(0,max_time,time_bin_size)

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
    
    return np.mean(accs), ttc, spk_hidden, s_out_rec