import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import set_random_seed
from utils_dataset import load_MNIST, Autoencoder_linear, load_dataset
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from torchvision.datasets import MNIST
import seaborn as sns

n_samples_to_plot = 5
gain = 0.02
fig, axs = plt.subplots(2,n_samples_to_plot)

generator = set_random_seed(42, add_generator=True)

# Full resolution MNIST:
dict_dataset = load_dataset('MNIST',
                            batch_size=10,
                            stim_len_sec=3,
                            dt_sec=0.001,
                            v_max=0.2,
                            generator=generator,
                            add_noise=True,
                            return_fft=False,
                            n_samples_train=-1,
                            n_samples_test=-1,
                            shuffle=True,
                            compressed=False,
                            gain=gain)

examples = enumerate(dict_dataset['train_loader'])
batch_idx, (example_data, example_targets) = next(examples)
batch_size, n_features, n_inputs = example_data.shape
print(batch_size)
print(n_features)
print(n_inputs)
for i in range(n_samples_to_plot):
    sns.heatmap(torch.transpose(example_data[i,:,:],1,0), ax=axs[0,i])
    axs[0,i].set_xlabel('Time bins')
    axs[0,i].set_ylabel('Channels')
    if i==0:
        axs[0,i].set_title(example_targets[i].item())

generator = set_random_seed(42, add_generator=True)

# Downscaled MNIST:
dict_dataset = load_dataset('MNIST',
                            batch_size=10,
                            stim_len_sec=3,
                            dt_sec=0.001,
                            v_max=0.2,
                            generator=generator,
                            add_noise=True,
                            return_fft=False,
                            n_samples_train=-1,
                            n_samples_test=-1,
                            shuffle=True,
                            compressed=True,
                            encoder_model='./data/784MNIST_2_24MNIST.pt',
                            gain=0.25)

examples = enumerate(dict_dataset['train_loader'])
batch_idx, (example_data, example_targets) = next(examples)
batch_size, n_features, n_inputs = example_data.shape
print(batch_size)
print(n_features)
print(n_inputs)
plt.figure()
sns.heatmap(example_data[0,:,:])
plt.show()

for i in range(n_samples_to_plot):
    sns.heatmap(torch.transpose(example_data[i,:,:],1,0), ax=axs[1,i])
    axs[1,i].set_xlabel('Time bins')
    axs[1,i].set_ylabel('Channels')
    if i==0:
        axs[1,i].set_title(example_targets[i].item())

fig.set_size_inches(20,10)
plt.subplots_adjust(hspace=0.3, wspace=0.2)
fig.savefig('./dataset_analysis/MNIST/fig/24_compressed.pdf', format='pdf')