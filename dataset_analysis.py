import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
import seaborn as sns
import numpy as np
from datasets import load_data
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from numpy.fft import rfft, rfftfreq
import tqdm
import os
from scipy import signal
import json
from torchvision.datasets import MNIST


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
firing_mode_dict = {
    "FA": {"a": 5, "A1": 0, "A2": 0},
    "SA": {"a": 0, "A1": 0, "A2": 0},
    "MIX": {"a": 5, "A1": 5, "A2": -0.3},
}


def sweep_classifier_hist(data,centers,spans,labels,labels_ascii_unique,name,cmap='Blues',folder_fig = '',folder_data = '',load=False,n_steps=100):
    if not load:
        matrix = np.zeros((len(centers),len(spans)))
        sweep = tqdm.tqdm(total=len(centers)*len(spans), desc=f"{name[0].upper() + name[1:]} Sweeping",position=0,leave=True)
        epochs = tqdm.trange(100, desc=f"Classifier",leave=False,position=1)
        for c_idx,center in enumerate(centers):
            for s_idx,span in enumerate(spans):
                acc = classify_hist(data.numpy(),center,span,labels,labels_ascii_unique,epochs,bins_n=n_steps)
                matrix[c_idx,s_idx] = acc
                sweep.update()
                np.save(os.path.join(folder_data,
                                     f'{name}_accuracy_c{centers[0]}_{centers[-1]}_{centers[1] - centers[0]}_s{spans[0]}_{spans[-1]}_{spans[1] - spans[0]}.npy'),
                        matrix)

    else:
        matrix = np.load(os.path.join(folder_data,f'{name}_accuracy_c{centers[0]}_{centers[-1]}_{centers[1]-centers[0]}_s{spans[0]}_{spans[-1]}_{spans[1]-spans[0]}.npy'))
    plt.imshow(matrix*100,aspect='auto',origin='lower',cmap=cmap)
    max = np.unravel_index(np.argmax(matrix),matrix.shape)
    which_decimal_c = np.max([len(str(int(0.99/(centers[1]-centers[0])))),len(str(int(0.99/(centers[0]))))])
    which_decimal_s = np.max([len(str(int(0.99/(spans[1]-spans[0])))),len(str(int(0.99/(spans[0]))))])
    plt.xticks(np.arange(len(spans)),np.round(spans,which_decimal_s),rotation=90)
    plt.yticks(np.arange(len(centers)),np.round(centers,which_decimal_c))
    plt.xlabel('Span')
    plt.ylabel('Center')
    plt.colorbar()
    plt.title(f'{name[0].upper() + name[1:]} Accuracy(%). Max:{(matrix.max()*100).astype(int)}%@({np.round(centers[max[0]],which_decimal_c)},{np.round(spans[max[1]],which_decimal_s)})')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_fig,f'{name}_accuracy_c{centers[0]}_{centers[-1]}_{np.round(centers[1]-centers[0],which_decimal_c)}_s{spans[0]}_{spans[-1]}_{np.round(spans[1]-spans[0],which_decimal_s)}.pdf'))
    return centers[max[0]],spans[max[1]],matrix.max()*100

def plot_best_hist(data,center,span,labels,label_ascii_unique,name,bins_n = 100,folder_fig=''):
    plt.figure()
    idx_letters, indices = np.unique(labels, return_index=True)
    colors = sns.color_palette("husl", len(idx_letters))
    sel_coll = []
    fig1, axs1 = plt.subplots(1, 1)
    for letter in range(len(label_ascii_unique)):
        idx_to_plot = np.where(labels == letter)[0]
        xhere = data[idx_to_plot, :, :]
        xhere = xhere.permute(0, 2, 1)
        xhere = xhere.flatten(0, 1)
        # axs1[0].plot(xhere.T, color=tuple(np.array(colors[letter])*1.1), label=label_ascii_unique[letter], alpha=0.01,zorder=letter)
        xsel = xhere[:, 100:250]
        xsel[xsel == 0] = torch.nan
        xhere[xhere == 0] = torch.nan
        sel_avg = torch.nanmean(xsel, dim=1)
        # axs1[0].plot(torch.nanmean(xhere, dim=0), color=colors[letter], zorder=letter + len(label_ascii_unique))

        sel_coll.append(sel_avg)
        bins, edges = np.histogram(sel_avg, bins=100, range=(center-span/2, center+span/2))
        axs1.bar(x=edges[:-1], height=bins / bins.max(), color=colors[letter], label=label_ascii_unique[letter],
                    alpha=0.5, zorder=letter, bottom=letter * 1.1, width=edges[1] - edges[0])
    fig1.savefig(os.path.join(folder_fig,name+'hist.png'))

def classify_hist(data,center,span,labels,labels_ascii_unique,epochs= None,bins_n=100):
    bins_coll = []
    # for letter in range(len(labels_ascii_unique)):
    for trial in range(data.shape[0]):
        datax = data[trial].flatten()
        # datax[datax == 0] = torch.nan
        bins, edges = np.histogram(datax,bins=bins_n,range=(center-span/2, center+span/2))

        bins_coll.append(bins)
    bins_coll = np.array(bins_coll)
    loss_coll,acc_coll = classifier(bins_coll,labels,epochs=epochs)
    return acc_coll[-1]
def classifier(data,labels,epochs=None):
    # data is a numpy array [trials, channels]
    # labels is a numpy array [trials]
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42,stratify=labels,shuffle=True)
    train_ds = TensorDataset(torch.from_numpy(x_train).to(device), y_train.to(device))
    test_ds = TensorDataset(torch.from_numpy(x_test).to(device), y_test.to(device))
    train_dl = DataLoader(train_ds, batch_size=1000, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1000, shuffle=True)
    classifier = nn.Sequential(
        nn.Linear(data.shape[1], len(np.unique(labels))),
    ).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    if epochs is None:
        epochs = trange(100, desc=f"Classifier")
    else:
        epochs.reset()
    batches = tqdm.tqdm(train_dl, desc="Epoch", disable=True)
    loss = nn.CrossEntropyLoss()
    loss_coll = []
    acc_coll = []
    for epoch in range(100):
        loss_list = []
        acc_list = []
        batches.reset()
        batches.set_description('Training')
        for batch_idx, (data, target) in enumerate(train_dl):
            data = data.float()
            target = target.long()
            out = classifier(data)
            loss_val = loss(out, target)
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss_val.item())
            batches.update()
        batches.reset(total=len(test_dl))
        batches.set_description('Testing')
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_dl):
                data = data.float()
                target = target.long()
                out = classifier(data)
                acc = torch.mean((torch.argmax(out, dim=1) == target).to(torch.int16), dtype=torch.float)
                acc_list.append(acc.item())
                batches.update()
        loss_coll.append(np.mean(loss_list))
        acc_coll.append(np.mean(acc_list))
        epochs.set_postfix_str(f"Loss: {np.mean(loss_list):.3f}, Acc: {np.mean(acc_list):.3f}")
        epochs.update()
    return loss_coll,acc_coll
def plt_amplitude(data,labels,label_ascii,folder_fig = ''):
    plt.figure()
    idx_letters, indices = np.unique(labels, return_index=True)
    colors = sns.color_palette("husl", len(idx_letters))
    label_ascii_unique = np.unique(label_ascii)
    sel_coll = []
    fig1, axs1 = plt.subplots(1, 1)
    for letter in range(len(label_ascii_unique)):
        idx_to_plot = np.where(labels == letter)[0]
        xhere = data[idx_to_plot, :, :]
        xhere = xhere.permute(0, 2, 1)
        xhere = xhere.flatten(0, 1)
        # axs1[0].plot(xhere.T, color=tuple(np.array(colors[letter])*1.1), label=label_ascii_unique[letter], alpha=0.01,zorder=letter)
        xsel = xhere[:, 100:250]
        xsel[xsel == 0] = torch.nan
        xhere[xhere == 0] = torch.nan
        sel_avg = torch.nanmean(xsel, dim=1)
        axs1.plot(torch.nanmean(xhere, dim=0), color=colors[letter], zorder=letter + len(label_ascii_unique))
        #
        # sel_coll.append(sel_avg)
        # bins, edges = np.histogram(sel_avg, bins=100, range=(0.1, 4))
        # axs1[1].bar(x=edges[:-1], height=bins / bins.max(), color=colors[letter], label=label_ascii_unique[letter],
        #             alpha=0.5, zorder=letter, bottom=letter * 1.1, width=edges[1] - edges[0])
    fig1.savefig(os.path.join(folder_fig,'amplitude.png'))
    # plt.show()
#
def extract_interval(data,freqs,samples_n,center,span):
    frange = np.linspace(center-span/2,center+span/2,samples_n)
    data_f = np.zeros(samples_n)
    for f in range(len(frange)-1):
        idx = np.where(np.logical_and(freqs>=frange[f],freqs<frange[f+1]))[0]
        if np.isnan(np.mean(data[idx])):
            data_f[f] = 0
        else:
            data_f[f] = np.mean(data[idx])

    return data_f

def sweep_classifier_fft(data,centers,spans,labels,labels_ascii_unique,name,cmap='Blues',folder_fig = '', folder_data = '', sample_size=10,load=False):
    if not load:
        matrix = np.zeros((len(centers),len(spans)))
        sweep = tqdm.tqdm(total=len(centers)*len(spans), desc=f"{name[0].upper() + name[1:]} Sweeping",position=0,leave=True)
        epochs = tqdm.trange(100, desc=f"Classifier",leave=False,position=1)
        for c_idx,center in enumerate(centers):
            for s_idx,span in enumerate(spans):
                acc = classify_fft(data.numpy(),center,span,labels,labels_ascii_unique,epochs,sample_size=sample_size)
                matrix[c_idx,s_idx] = acc
                sweep.update()
        np.save(os.path.join(folder_data,f'{name}_accuracy_c{centers[0]}_{centers[-1]}_{centers[1]-centers[0]}_s{spans[0]}_{spans[-1]}_{spans[1]-spans[0]}.npy'),matrix)
    else:
        matrix = np.load(os.path.join(folder_data,f'{name}_accuracy_c{centers[0]}_{centers[-1]}_{centers[1]-centers[0]}_s{spans[0]}_{spans[-1]}_{spans[1]-spans[0]}.npy'))
    plt.imshow(matrix*100,aspect='auto',origin='lower',cmap=cmap)
    max = np.unravel_index(np.argmax(matrix),matrix.shape)
    which_decimal_c = np.max([int(0.99/(centers[1]-centers[0])),int(0.99/centers[0])])
    which_decimal_s = np.max([int(0.99/(spans[1]-spans[0])),int(0.99/spans[0])])
    plt.xticks(np.arange(len(spans)),np.round(spans,which_decimal_s).astype(int))
    plt.yticks(np.arange(len(centers)),np.round(centers,which_decimal_c))
    plt.xlabel('Span')
    plt.ylabel('Center')
    plt.colorbar()
    plt.title(f'{name[0].upper() + name[1:]} Accuracy(%). Max:{(matrix.max()*100).astype(int)}%@({np.round(centers[max[0]],which_decimal_c)},{np.round(spans[max[1]],which_decimal_s)})')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_fig,f'{name}_accuracy_c{centers[0]}_{centers[-1]}_{np.round(centers[1]-centers[0],3)}_s{spans[0]}_{spans[-1]}_{np.round(spans[1]-spans[0],3)}.pdf'))
    return centers[max[0]], spans[max[1]], matrix.max() * 100
def classify_fft(data,center,span,labels,labels_ascii_unique,epochs= None,sample_size=10):
    n_samples = data.shape[1]  # number of time steps
    dt = (1 / 100.0)
    b, a = signal.butter(3, 0.1, 'high')
    frange = (center - span / 2, center + span / 2)
    data_fft_coll = []
    for trial in range(data.shape[0]):
        x = signal.filtfilt(b, a, data[trial], axis=1)
        xf = rfftfreq(n_samples, dt)
        yf = rfft(x, axis=0)
        yf = np.abs(yf)
        data_f = extract_interval(yf,xf,sample_size,center,span)
        data_fft_coll.append(data_f)
    data_fft_coll = np.array(data_fft_coll)
    loss_coll,acc_coll = classifier(data_fft_coll,labels,epochs=epochs)
    return acc_coll[-1]
def plt_frequency(data,labels,label_ascii,folder_fig=''):
    plt.figure()
    dt = (1 / 100.0)
    idx_letters, indices = np.unique(labels, return_index=True)
    b, a = signal.butter(3, 0.1, 'high')
    n_samples = data.shape[1]  # number of time steps
    colors = sns.color_palette("husl", len(idx_letters))
    label_ascii_unique = np.unique(label_ascii)
    for letter in range(np.unique(labels).shape[0]):
        # for letter in range(19,21):
        idx_to_plot = np.where(labels == letter)[0]
        x = data[idx_to_plot, :, :]
        x = signal.filtfilt(b, a, x, axis=1)
        xf = rfftfreq(n_samples, dt)
        yf = rfft(x, axis=1)
        y = np.mean(np.mean(np.abs(yf), axis=2), axis=0)
        y = y / np.max(y)
        plt.plot(xf, y + 0.1 * (len(idx_letters) - letter), 'k', zorder=(letter) * 2 + 1)
        # color area under plot
        plt.fill_between(xf, y + 0.1 * (len(idx_letters) - letter), alpha=1, color=colors[letter],
                         label=label_ascii_unique[letter], zorder=(letter) * 2)
    plt.yticks([])
    plt.legend(fancybox=True, shadow=True, ncol=2, bbox_to_anchor=(1.05, 1), title='Braille Letter')
    plt.tight_layout()
    # remove top and right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlabel('Frequency (Hz)')
    plt.savefig(os.path.join(folder_fig,'Frequency.pdf'), format='pdf', dpi=300)

def do_analysis(data,labels,label_ascii,folder_fig,folder_data,args):
    label_ascii_unique = np.unique(label_ascii)
    opt = {}
    found = False
    # ------------------------------------------------------------------------------------------ Amplitude:
    if 'amplitude' in args.analysis:
        found = True
        print('Amplitude')
        plt_amplitude(data, labels, label_ascii, folder_fig)
        plt.figure()
        c, s, a = sweep_classifier_hist(data=data,
                                        centers=np.linspace(0.5, 100, 10),
                                        spans=np.linspace(0.5, 10, 10),
                                        n_steps=100,
                                        labels=labels,
                                        labels_ascii_unique=label_ascii_unique,
                                        name='amplitude',
                                        cmap='Blues',
                                        folder_fig=folder_fig,
                                        folder_data=folder_data,
                                        load=args.load)
        plot_best_hist(data=data,
                          center=c,
                          span=s,
                          labels=labels,
                          label_ascii_unique=label_ascii_unique,
                          name='amplitude',
                          folder_fig=folder_fig,
                          )
        opt['amplitude'] = {'center': c,
                            'span': s,
                            'n_steps': 10,
                            'acc': a}
    # ------------------------------------------------------------------------------------------ Frequency:
    if 'frequency' in args.analysis:
        found = True
        print('Frequency')
        plt_frequency(data, labels, label_ascii, folder_fig=folder_fig)
        plt.figure()
        c, s, a = sweep_classifier_fft(data=data,
                                       centers=np.linspace(10, 20, 10),
                                       spans=np.linspace(10, 20, 10),
                                       labels=labels,
                                       labels_ascii_unique=label_ascii_unique,
                                       name='frequency',
                                       cmap='Reds',
                                       folder_fig=folder_fig,
                                       folder_data=folder_data,
                                       sample_size=10,
                                       load=args.load)
        opt['frequency'] = {'center': c,
                            'span': s,
                            'n_steps': 10,
                            'acc': a}
    # ------------------------------------------------------------------------------------------ Slope :
    if 'slope' in args.analysis:
        found = True
        print('Slope')
        slope = torch.diff(data, dim=1)
        plt.figure()
        c, s, a = sweep_classifier_hist(data=slope,
                                        centers=np.linspace(0.5, 5, 10),
                                        spans=np.linspace(0.5, 1, 10),
                                        n_steps=100,
                                        labels=labels,
                                        labels_ascii_unique=label_ascii_unique,
                                        name='slope',
                                        cmap='Greens',
                                        folder_fig=folder_fig,
                                        folder_data=folder_data,
                                        load=args.load)
        plot_best_hist(data=slope,
                          center=c,
                          span=s,
                          labels=labels,
                          label_ascii_unique=label_ascii_unique,
                          name='slope',
                          folder_fig=folder_fig,
                          )
        opt['slope'] = {'center': c,
                        'span': s,
                        'n_steps': 10,
                        'acc': a}
    if found == False:
        raise ValueError('No analysis found')

    json.dump(opt, open(os.path.join(folder_data, 'opt.json'), 'w'))
def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    folder = Path('dataset_analysis')
    if 'Braille' in args.dataset:
        print('Braille')
        folder_run = Path(os.path.join(folder,'Braille'))
        folder_fig = folder_run.joinpath('fig')
        folder_data = folder_run.joinpath('data')
        folder_fig.mkdir(parents=True, exist_ok=True)
        folder_data.mkdir(parents=True, exist_ok=True)
        folder_fig = str(folder_fig)
        folder_data = str(folder_data)
        ###########################################
        ##                Dataset                ##
        ###########################################
        upsample_fac = 1
        gain = 10
        dt = (1 / 100.0) / upsample_fac
        file_name = "data/data_braille_letters_all.pkl"
        data, labels, _, _, _, _, label_ascii = load_data(file_name, upsample_fac,label_ascii=True)
        data *= gain
        do_analysis(data,labels,label_ascii,folder_fig,folder_data,args)
    if 'MNIST' in args.dataset:
        print('MNIST')
        folder_run = Path(os.path.join(folder,'MNIST'))
        folder_fig = folder_run.joinpath('fig')
        folder_data = folder_run.joinpath('data')
        folder_fig.mkdir(parents=True, exist_ok=True)
        folder_data.mkdir(parents=True, exist_ok=True)
        folder_fig = str(folder_fig)
        folder_data = str(folder_data)
        data = MNIST(
            root="data",
            train=True,
            download=True,
            )
        time_length = 300
        limited_samples = 2000
        data_MNIST = data.data[:limited_samples].flatten(start_dim=1,end_dim=2).unsqueeze(1).repeat(1, time_length, 1)
        data_MNIST = data_MNIST.to(torch.float) + torch.randint_like(data_MNIST,high=10)*0.02
        train_labels = data.targets[:limited_samples]
        do_analysis(data_MNIST,train_labels,train_labels,folder_fig,folder_data,args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser('dataset_analysis')
    parser.add_argument('--stim_length_sec', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--dt', type=float, default=0.001)
    parser.add_argument('--load',action='store_true')
    parser.add_argument('--dataset', type=str, default='Braille,MNIST')
    parser.add_argument('--analysis', type=str, default='amplitude,frequency,slope')
    args = parser.parse_args()
    if ',' in args.dataset:
        args.dataset = args.dataset.split(',')
    else:
        args.dataset = [args.dataset]
    if ',' in args.analysis:
        args.analysis = args.analysis.split(',')
    else:
        args.analysis = [args.analysis]
    main(args)
