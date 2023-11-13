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
import pandas as pd
import h5py
from utils_dataset import load_dataset, set_random_seed, extract_interval, extract_histogram, get_fft

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
firing_mode_dict = {
    "FA": {"a": 5, "A1": 0, "A2": 0},
    "SA": {"a": 0, "A1": 0, "A2": 0},
    "MIX": {"a": 5, "A1": 5, "A2": -0.3},
}
from scipy import stats
def corrfunc(x, y, **kws):
    r, p = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}, p = {:.2f}".format(r, p),
                xy=(.1, .98), xycoords=ax.transAxes)
class MNISTDataset_current(torch.utils.data.dataset.Dataset):
    """
    Load MNIST dataset as custom dataset generated by converting MNIST digits into input currents or into feature sets
    (frequency values, amplitudes and slopes).
    """

    def __init__(self, hdf5_file, device=None,gain=None):
        """
            :param data_file: Path to h5 file with dataset.
        """
        self.file = hdf5_file
        self.device = device
        if gain is None:
            self.gain = gain
        else:
            self.gain = 1
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

def set_random_seed(seed, add_generator=False, device=torch.device('cpu')):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if add_generator:
        generator = torch.Generator(device=device).manual_seed(seed)
        return generator
    else:
        return None
def sweep_classifier_hist(data,centers,spans,labels,labels_ascii_unique,name,cmap='Blues',folder_fig = '',folder_data = '',load=False,n_steps=100,datatype='',dataset='',data_dict=None):
    if not load:
        matrix = np.zeros((len(centers),len(spans)))
        sweep = tqdm.tqdm(total=len(centers)*len(spans), desc=f"{name[0].upper() + name[1:]} Sweeping",position=0,leave=True)
        epochs = tqdm.trange(100, desc=f"Classifier",leave=False,position=1)
        data_dict = {'center': [], 'span': [], 'accuracy': [], 'data_type': [], 'dataset': []}

        for c_idx,center in enumerate(centers):
            for s_idx,span in enumerate(spans):
                acc = classify_hist(data.numpy(),center,span,labels,labels_ascii_unique,epochs,bins_n=n_steps,data_dict=data_dict)
                matrix[c_idx,s_idx] = acc
                sweep.update()
                data_dict['center'].append(center)
                data_dict['span'].append(span)
                data_dict['accuracy'].append(acc)
                data_dict['data_type'].append(datatype)
                data_dict['dataset'].append(dataset)
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
    data_dict = pd.DataFrame(data_dict)
    data_dict.head()
    plt.figure()
    g = sns.PairGrid(data=data_dict[data_dict['data_type']==datatype], y_vars=["accuracy"], x_vars=["center", "span"], height=4)
    g.map(sns.regplot)
    g.map(corrfunc)
    g.fig.subplots_adjust(top=0.8) # adjust the Figure in rp
    N = len(centers)
    g.fig.suptitle(f'feature: {datatype}')
    plt.savefig(os.path.join(folder_fig,f'{name}_corr_c{centers[0]}_{centers[-1]}_{np.round(centers[1]-centers[0],which_decimal_c)}_s{spans[0]}_{spans[-1]}_{np.round(spans[1]-spans[0],which_decimal_s)}.pdf'))
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

def classify_hist(data,center,span,labels,labels_ascii_unique,epochs= None,bins_n=100,data_dict=None):
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
def plt_amplitude(dataloader,label_ascii,folder_fig = ''):
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
    data_f = np.zeros([data.shape[0],samples_n])
    # print(data.shape)
    for f in range(len(frange)-1):
        idx = np.where(np.logical_and(freqs>=frange[f],freqs<frange[f+1]))[0]
        if len(idx) == 0:
            data_f[:,f] = 0
        elif np.isnan(np.mean(np.array(data)[idx])):
            data_f[:,f] = 0
        else:
            for trial in range(len(data)):

                data_f[trial,f] = np.mean(data[trial,idx])

    return data_f

def sweep_classifier_fft(data,centers,spans,labels,labels_ascii_unique,name,cmap='Blues',folder_fig = '', folder_data = '', sample_size=10,load=False,dataset = ''):
    if not load:
        matrix = np.zeros((len(centers),len(spans)))
        sweep = tqdm.tqdm(total=len(centers)*len(spans), desc=f"{name[0].upper() + name[1:]} Sweeping",position=0,leave=True)
        epochs = tqdm.trange(100, desc=f"Classifier",leave=False,position=1)
        data_dict = {'center': [], 'span': [], 'accuracy': [], 'data_type': [], 'dataset': []}

        for c_idx,center in enumerate(centers):
            for s_idx,span in enumerate(spans):
                acc = classify_fft(data.numpy(),center,span,labels,labels_ascii_unique,epochs,sample_size=sample_size)
                matrix[c_idx,s_idx] = acc
                sweep.update()
                data_dict['center'].append(center)
                data_dict['span'].append(span)
                data_dict['accuracy'].append(acc)
                data_dict['data_type'].append('fft')
                data_dict['dataset'].append(dataset)
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
    data_dict = pd.DataFrame(data_dict)
    data_dict.head()
    plt.figure()
    g = sns.PairGrid(data=data_dict[data_dict['data_type'] == 'fft'], y_vars=["accuracy"], x_vars=["center", "span"],
                     height=4)
    g.map(sns.regplot)
    g.map(corrfunc)
    g.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
    N = len(centers)
    g.fig.suptitle(f'feature: fft')
    plt.savefig(os.path.join(folder_fig,
                             f'{name}_corr_c{centers[0]}_{centers[-1]}_{np.round(centers[1] - centers[0], which_decimal_c)}_s{spans[0]}_{spans[-1]}_{np.round(spans[1] - spans[0], which_decimal_s)}.pdf'))
    return centers[max[0]], spans[max[1]], matrix.max() * 100
def classify_fft(data,center,span,labels,labels_ascii_unique,epochs= None,sample_size=10):
    n_samples = data.shape[1]  # number of time steps
    dt = (1 / 100.0)
    b, a = signal.butter(3, 0.1, 'high')
    frange = (center - span / 2, center + span / 2)
    data_fft_coll = []
    for trial in range(data.shape[0]):
        # plt.plot(data[trial])
        # print(data[trial].shape)
        # plt.figure()
        x = signal.filtfilt(b, a, data[trial], axis=0)
        xf = rfftfreq(n_samples, dt)
        yf = rfft(x, axis=0)
        yf = np.abs(yf)
        # plt.plot(xf, yf)
        # plt.title('FFT after HP')
        # plt.show()
        # raise ValueError
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
def plt_signal(dict_dataset,dataset_name,folder_fig=''):
    data_coll = []
    for i, (data, target) in enumerate(dict_dataset['train_loader']):
        data = extract_feature(data, dict_dataset, 'mean_amplitude', 0, 0, 0, args, i,target=target)
        data_coll.append(data)
    data_lump = {}
    for key in range(dict_dataset['n_classes']):
        for data in data_coll:
            if key not in data_lump.keys():
                data_lump[key] = []
            if key in data.keys():
                data_lump[key].append(data[key])
    plt.figure()
    for key in data_lump.keys():
        data_lump[key] = torch.mean(torch.stack(data_lump[key]),dim=0)
        plt.plot(data_lump[key][:,:].cpu().numpy(),color=sns.color_palette("husl", dict_dataset['n_classes'])[key])
    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], color=sns.color_palette("husl", dict_dataset['n_classes'])[key], lw=4) for key in data_lump.keys()]
    labels = [str(key) for key in data_lump.keys()]
    cols = len(labels)//10
    plt.legend(custom_lines, [str(key) for key in data_lump.keys()],ncol=cols)
    # plt.legend()
    plt.title('Mean Amplitude '+dataset_name)
    plt.savefig(os.path.join(folder_fig,'mean_amplitude.png'))
def plt_fft(dict_dataset,dataset_name,folder_fig='',f_min=5):
    data_coll = []
    for i, (data, target) in enumerate(dict_dataset['train_loader']):
        data,xf = extract_feature(data, dict_dataset, 'mean_fft', 0, 0, 0, args, i,target=target)
        data_coll.append(data)
    data_lump = {}
    for key in range(dict_dataset['n_classes']):
        for data in data_coll:
            if key not in data_lump.keys():
                data_lump[key] = []
            if key in data.keys():
                data_lump[key].append(data[key])
    plt.figure()
    xf_where = xf >= f_min
    for key in data_lump.keys():
        data_lump[key] = torch.mean(torch.stack(data_lump[key]),dim=0)
        plt.plot(xf[xf_where],data_lump[key][xf_where,:].cpu().numpy(),color=sns.color_palette("husl", dict_dataset['n_classes'])[key])
    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], color=sns.color_palette("husl", dict_dataset['n_classes'])[key], lw=4) for key in data_lump.keys()]
    labels = [str(key) for key in data_lump.keys()]
    cols = len(labels)//10
    plt.legend(custom_lines, [str(key) for key in data_lump.keys()],ncol=cols)
    # plt.legend()
    plt.title('Mean FFT '+dataset_name)
    plt.savefig(os.path.join(folder_fig,'mean_fft.png'))
def plt_best(opt,dict_dataset,data_type,xf=None,dataset_name = '',folder_fig=''):
    center = opt[data_type]['center']
    span = opt[data_type]['span']
    data_coll = {}
    for label in range(dict_dataset['n_classes']):
        data_coll[label] = []
    for i, (data, target) in enumerate(dict_dataset['train_loader']):
        data = extract_feature(data, dict_dataset, data_type, center, span, args.bins, args, i, xf=xf)
        data = torch.tensor(data).to(torch.float)
        for label in range(dict_dataset['n_classes']):
            data_coll[label].append(data[target==label])
    plt.figure()
    for label in range(dict_dataset['n_classes']):
        # print(data_coll[label])

        data_coll[label] = torch.mean(torch.concatenate(data_coll[label]),dim=0)
        plt.plot(data_coll[label].cpu().numpy(),color=sns.color_palette("husl", dict_dataset['n_classes'])[label])
    plt.title('Best '+data_type + ' '+dataset_name)
    plt.savefig(os.path.join(folder_fig,'best_'+data_type+'.png'))
    # plt.show()

def extract_feature(data, dict_dataset, data_type,center,span,bins,args, i, xf=None,target= None):
    assert data.shape[1] == dict_dataset['n_features']
    assert data.shape[2] == dict_dataset['n_inputs']
    # print(data_type)
    # print(data.shape)
    if data_type == 'frequency':
        xf,data = get_fft(data, dt=1/100)
        # print(data.shape)
        data = extract_interval(data, xf, bins, center, span)
        # print(data.shape)
        assert data.shape[1] == bins

    elif data_type == 'amplitude':
        data = extract_histogram(data, bins, center, span)
        assert data.shape[1] == bins

    elif data_type == 'slope':
        data = torch.diff(data, dim=1)
        data = extract_histogram(data, bins, center, span)
        assert data.shape[1] == bins

    elif data_type == 'mean_amplitude':
        # print(data.shape)
        data_mean = {}
        for t in target.unique():
            data_s = data[target == t]
            # print(data_s.shape)

            data_mean[t.item()] = torch.mean(data_s, dim=0)
        # print(data_mean)
        return data_mean
    elif data_type == 'mean_fft':
        data_mean = {}
        for t in target.unique():
            data_s = data[target == t]
            data_s_fft = torch.abs(torch.fft.rfft(data_s, dim=1))
            xf = rfftfreq(data_s.shape[1], 1/100)
            data_mean[t.item()] = torch.mean(data_s_fft,dim=0)
        return data_mean,xf


    return data
def classifier_processed(dict_dataset,epochs,data_type,center,span,args,xf=None):
    classifier = nn.Sequential(
        nn.Linear(args.bins, dict_dataset['n_classes']),
    ).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    if epochs is None:
        epochs = trange(args.epochs_n, desc=f"Classifier {data_type} {center} {span}")
    else:
        epochs.set_description(f"Classifier {data_type} {center} {span}")
        epochs.reset()
    batches = tqdm.tqdm(dict_dataset['train_loader'], desc="Batches", disable=True)
    loss = nn.CrossEntropyLoss()
    loss_coll = []
    acc_coll = []
    for epoch in epochs:
        loss_list = []
        acc_list = []
        batches.reset()
        batches.set_description('Training')
        for i, (data, target) in enumerate(dict_dataset['train_loader']):
            # data *= dict_dataset['train_loader'].gain
            # print('data_before',data.shape)
            data = extract_feature(data, dict_dataset, data_type, center, span, args.bins, args, i, xf=xf)
            data = torch.tensor(data).to(device)
            data = data.float()
            target = target.long()
            target = target.to(device)
            out = classifier(data)
            loss_val = loss(out, target)
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss_val.item())
            batches.update()
        batches.reset(total=len(dict_dataset['test_loader']))
        batches.set_description('Testing')
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dict_dataset['test_loader']):

                data = extract_feature(data, dict_dataset, data_type, center, span, args.bins, args, batch_idx, xf=xf)
                data = torch.tensor(data).to(device)
                data = data.float()
                target = target.long()
                target = target.to(device)
                out = classifier(data)
                acc = torch.mean((torch.argmax(out, dim=1) == target).to(torch.int16), dtype=torch.float)
                acc_list.append(acc.item())
                batches.update()
        loss_coll.append(np.mean(loss_list))
        acc_coll.append(np.mean(acc_list))
        epochs.set_postfix_str(f"Loss: {np.mean(loss_list):.3f}, Acc: {np.mean(acc_list):.3f}")
        epochs.update()
    return loss_coll, acc_coll
def do_analysis(dict_dataset,analysis,centers,spans,folder_fig='',folder_data='',dataset_name='',args=None):
    opt = {}

    epochs = None
    xf = rfftfreq(dict_dataset['n_time_steps'], dict_dataset['dt_sec'])

    for data_type in analysis:
        plot_dict = {'center': [], 'span': [], 'accuracy': [], 'data_type': [], 'dataset': [], 'sim_id': []}
        matrix = np.zeros((len(centers[data_type]),len(spans[data_type])))
        for c_idx,center in enumerate(centers[data_type]):
            for s_idx,span in enumerate(spans[data_type]):
                loss,acc = classifier_processed(dict_dataset, epochs, data_type, center, span, args, xf=xf)
                matrix[c_idx,s_idx] = np.mean(acc[-2:])
                plot_dict['center'].append([center]*len(acc))
                plot_dict['span'].append([span]*len(acc))
                plot_dict['accuracy'].append(acc)
                plot_dict['data_type'].append([data_type]*len(acc))
                plot_dict['dataset'].append([dataset_name]*len(acc))
                if args.sim_id>=0:
                    plot_dict['sim_id'].append([args.sim_id]*len(acc))
                    np.save(os.path.join(folder_data,f'{data_type}_accuracy_c{center}_s{span}_{args.sim_id}.npy'),plot_dict)
        if args.sim_id<0:
            np.save(os.path.join(folder_data,
                                 f'{data_type}_accuracy_c{centers[data_type][0]}_{centers[data_type][-1]}_{centers[data_type][1] - centers[data_type][0]}_s{spans[data_type][0]}_{spans[data_type][-1]}_{spans[data_type][1] - spans[data_type][0]}.npy'),
                    matrix)

def retrieve_analysis(analysis,centers,spans,folder_data='',args=None):
    plot_dicts = []
    for d_idx,data_type in enumerate(analysis):
        print(data_type)
        for c_idx,center in enumerate(centers[data_type]):
            for s_idx,span in enumerate(spans[data_type]):
                # print(data_type,center,span)
                id = s_idx*30 + c_idx*3 + d_idx
                # print(id)
                plot_dict = np.load(os.path.join(folder_data,f'{data_type}_accuracy_c{center}_s{span}_{id}.npy'),allow_pickle=True).item()
                plot_dicts.append(plot_dict)
    return plot_dicts
def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cmaps = {
        'frequency': 'Blues',
        'amplitude': 'Reds',
        'slope': 'Greens',
    }
    centers = {
        'frequency': np.linspace(10, 50, 10),
        'amplitude': np.linspace(0.5, 10, 10),
        'slope': np.linspace(0.5, 5, 10),
    }
    spans = {
        'frequency': np.linspace(1, 10, 10),
        'amplitude': np.linspace(0.5, 10, 10),
        'slope': np.linspace(0.5, 1, 10),
    }
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    generator = set_random_seed(args.seed, add_generator=True, device='cpu')
    folder = Path('dataset_analysis_hb')
    if (args.sim_id >= 0) & (args.load == False):
        # folder = folder.joinpath(f'sim_id_{args.sim_id}')
        # folder.mkdir(parents=True, exist_ok=True)
        sim_id_analysis = args.sim_id % len(args.analysis)
        remaining = args.sim_id // len(args.analysis)
        sim_id_center = remaining % len(centers[args.analysis[sim_id_analysis]])
        sim_id_span = args.sim_id // len(centers[args.analysis[sim_id_analysis]]) // len(args.analysis)
        # print(sim_id_analysis, sim_id_center, sim_id_span)
        analysis = [args.analysis[sim_id_analysis]]
        centers = {analysis[0]:[centers[analysis[0]][sim_id_center]]}
        spans = {analysis[0]:[spans[analysis[0]][sim_id_span]]}
    else:
        analysis = args.analysis
    for dataset in args.dataset:
        if 'Braille' == dataset:
            print('Braille')
            folder_run = Path(os.path.join(folder,'Braille'))
            folder_fig = folder_run.joinpath('fig')
            folder_data = folder_run.joinpath('data')
            folder_fig.mkdir(parents=True, exist_ok=True)
            folder_data.mkdir(parents=True, exist_ok=True)
            folder_fig = str(folder_fig)
            folder_data = str(folder_data)
            dict_dataset = load_dataset('Braille',
                                        batch_size=args.batch_size,
                                        generator=generator,
                                        upsample_fac=1,
                                        data_path="data/data_braille_letters_all.pkl",
                                        return_fft=False,
                                        sampling_freq_hz=100.0,
                                        v_max=-1,
                                        shuffle=True,
                                        gain = args.gain_Braille)
            if args.load == False:
                do_analysis(dict_dataset,analysis,centers,spans,folder_fig=folder_fig,folder_data=folder_data,args=args,dataset_name='Braille')



        elif 'MNIST' == dataset:
            print('MNIST')
            folder_run = Path(os.path.join(folder,'MNIST'))

            folder_fig = folder_run.joinpath('fig')
            folder_data = folder_run.joinpath('data')
            folder_fig.mkdir(parents=True, exist_ok=True)
            folder_data.mkdir(parents=True, exist_ok=True)
            folder_fig = str(folder_fig)
            folder_data = str(folder_data)
            dict_dataset = load_dataset('MNIST',
                                        batch_size=args.batch_size,
                                        stim_len_sec=3,
                                        dt_sec=1/100,
                                        v_max=0.2,
                                        generator=generator,
                                        add_noise=True,
                                        return_fft=False,
                                        n_samples_train=6480,
                                        n_samples_test=1620,
                                        shuffle=True,
                                        gain = args.gain_MNIST)
            if args.load == False:
                do_analysis(dict_dataset,analysis,centers,spans,folder_fig=folder_fig,folder_data= folder_data, args=args,dataset_name='MNIST')
        else:
            raise ValueError('dataset not found')
        if args.load:
            plt_signal(dict_dataset,dataset,folder_fig=folder_fig)
            plt_fft(dict_dataset,dataset,folder_fig=folder_fig)
            pd_datasets = []

            folder_run = folder.joinpath(dataset)
            folder_data = folder_run.joinpath('data')

            matrixes = retrieve_analysis(analysis,centers,spans,folder_data=str(folder_data),args=args)
            df1 = pd.DataFrame.from_dict(matrixes[0])
            for matrix in matrixes[1:]:
                df2 = pd.DataFrame.from_dict(matrix)
                df1=pd.concat([df1, df2])
            pd_datasets.append(df1)

            plot_dict = pd.concat(pd_datasets)
            plot_dict.head()

            folder_run = folder.joinpath(dataset)
            folder_fig = folder_run.joinpath('fig')
            folder_fig.mkdir(parents=True, exist_ok=True)
            folder_data = folder_run.joinpath('data')
            opt = {}
            for data_type in analysis:
                plot_dict_sel = plot_dict[(plot_dict['data_type'] == data_type) & (plot_dict['dataset']==dataset)]
                # plt.figure()
                # g = sns.PairGrid(data=plot_dict_sel, y_vars=["accuracy"],
                #                  x_vars=["center", "span"], height=4)
                # g.map(sns.regplot)
                # g.map(corrfunc)
                # g.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
                # N = len(centers[data_type])
                # g.fig.suptitle(f'dataset {dataset}, feature: {data_type}')
                which_decimal_c = np.max(
                    [len(str(int(0.99 / (centers[data_type][1] - centers[data_type][0])))),
                     len(str(int(0.99 / (centers[data_type][0]))))])
                which_decimal_s = np.max(
                    [len(str(int(0.99 / (spans[data_type][1] - spans[data_type][0])))),
                     len(str(int(0.99 / (spans[data_type][0]))))])
                # print(folder_fig)
                # plt.savefig(os.path.join(folder_fig,
                #                          f'{data_type}_corr_c{centers[data_type][0]}_{centers[data_type][-1]}_{np.round(centers[data_type][1] - centers[data_type][0], which_decimal_c)}_s{spans[data_type][0]}_{spans[data_type][-1]}_{np.round(spans[data_type][1] - spans[data_type][0], which_decimal_s)}.pdf'))
                # plt.savefig(os.path.join(folder_fig,
                #                          f'{data_type}_corr_c{centers[data_type][0]}_{centers[data_type][-1]}_{np.round(centers[data_type][1] - centers[data_type][0], which_decimal_c)}_s{spans[data_type][0]}_{spans[data_type][-1]}_{np.round(spans[data_type][1] - spans[data_type][0], which_decimal_s)}.png'))
                # # plt.close()
                plt.figure()
                plot_dict_hm = plot_dict_sel.pivot(index="center", columns="span", values="accuracy")
                plot_dict_hm_np = np.array(plot_dict_hm)
                max_here = np.unravel_index(np.argmax(plot_dict_hm_np), plot_dict_hm_np.shape)
                sns.heatmap(plot_dict_hm)
                plt.title(f'dataset {dataset}, feature: {data_type}')
                plt.xticks(np.arange(len(spans[data_type])), np.round(spans[data_type], which_decimal_s))
                plt.yticks(np.arange(len(centers[data_type])), np.round(centers[data_type], which_decimal_c))
                plt.savefig(os.path.join(folder_fig,
                                         f'{data_type}_hm_c{centers[data_type][0]}_{centers[data_type][-1]}_{np.round(centers[data_type][1] - centers[data_type][0], which_decimal_c)}_s{spans[data_type][0]}_{spans[data_type][-1]}_{np.round(spans[data_type][1] - spans[data_type][0], which_decimal_s)}.pdf'))
                plt.savefig(os.path.join(folder_fig,
                                         f'{data_type}_hm_c{centers[data_type][0]}_{centers[data_type][-1]}_{np.round(centers[data_type][1] - centers[data_type][0], which_decimal_c)}_s{spans[data_type][0]}_{spans[data_type][-1]}_{np.round(spans[data_type][1] - spans[data_type][0], which_decimal_s)}.png'))

                opt[data_type] = {'center': centers[data_type][max_here[0]],
                                  'span': spans[data_type][max_here[1]],
                                  'n_steps': 10,
                                  'acc': plot_dict_hm_np.max() * 100}
                plt_best(opt,dict_dataset,data_type,dataset_name = dataset,folder_fig=folder_fig)

            json.dump(opt, open(os.path.join(folder_data, 'opt.json'), 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('dataset_analysis')
    parser.add_argument('--stim_length_sec', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--dt', type=float, default=0.001)
    parser.add_argument('--load',action='store_true')
    parser.add_argument('--dataset', type=str, default='Braille,MNIST')
    parser.add_argument('--analysis', type=str, default='amplitude,frequency,slope')
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--epochs_n',  type=int, default=2)
    parser.add_argument('--bins',  type=int, default=100)
    parser.add_argument('--sim_id',  type=int, default=-1)
    parser.add_argument('--stim_len_sec',  type=int, default=300)
    parser.add_argument('--gain_Braille', type=float, default=10)
    parser.add_argument('--gain_MNIST', type=float, default=0.02)
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
