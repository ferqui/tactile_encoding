import matplotlib.pyplot as plt

from utils_dataset import set_random_seed, MNISTDataset_features, MNISTDataset_current
from torch import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import tqdm
import os
import torch
from tqdm import trange
import argparse
import h5py
import torch.nn as nn
from utils_dataset import extract_interval
import json


def train_classifier(train_dl, test_dl, n_epochs, sample_size, device):
    classifier = nn.Sequential(nn.Linear(sample_size, train_dl.dataset.n_classes, )).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    epochs = tqdm.trange(n_epochs, desc=f"Classifier", leave=False, position=1)

    batches = tqdm.tqdm(train_dl, desc="Epoch", disable=True)
    loss = nn.CrossEntropyLoss()
    loss_coll = []
    acc_coll = []

    for epoch in range(n_epochs):
        loss_list = []
        acc_list = []
        batches.reset()
        batches.set_description('Training')

        for batch_idx, (data, target) in enumerate(train_dl):
            target = target.squeeze().long()
            out = classifier(data.float())
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
                # if epoch == (n_epochs - 1):
                #     list_acc_last_epoch.append(acc.tolist())
                batches.update()

        loss_coll.append(np.mean(loss_list))
        acc_coll.append(np.mean(acc_list))
        epochs.set_postfix_str(f"Loss: {np.mean(loss_list):.3f}, Acc: {np.mean(acc_list):.3f}")
        epochs.update()

    return acc_list  # return test accuracy of last training epoch, last batch


# def sweep_classifier_fft(test_dl, train_dl, n_epochs, name, cmap='Blues', folder_fig='', folder_data='', sample_size=10,
#                          load=False):
#     if not load:
#         matrix = np.zeros((len(centers), len(spans)))
#         sweep = tqdm.tqdm(total=len(centers) * len(spans), desc=f"{name[0].upper() + name[1:]} Sweeping", position=0,
#                           leave=True)
#         epochs = tqdm.trange(n_epochs, desc=f"Classifier", leave=False, position=1)
#         for c_idx, center in enumerate(centers):
#             for s_idx, span in enumerate(spans):
#                 acc = train_classifier(center, span, test_dl, train_dl, n_epochs, sample_size)
#                 matrix[c_idx, s_idx] = acc
#                 sweep.update()
#         np.save(os.path.join(folder_data,
#                              f'{name}_accuracy_c{centers[0]}_{centers[-1]}_{centers[1] - centers[0]}_s{spans[0]}_{spans[-1]}_{spans[1] - spans[0]}.npy'),
#                 matrix)
#     else:
#         matrix = np.load(os.path.join(folder_data,
#                                       f'{name}_accuracy_c{centers[0]}_{centers[-1]}_{centers[1] - centers[0]}_s{spans[0]}_{spans[-1]}_{spans[1] - spans[0]}.npy'))
#     plt.imshow(matrix * 100, aspect='auto', origin='lower', cmap=cmap)
#     max = np.unravel_index(np.argmax(matrix), matrix.shape)
#     which_decimal_c = np.max([int(0.99 / (centers[1] - centers[0])), int(0.99 / centers[0])])
#     which_decimal_s = np.max([int(0.99 / (spans[1] - spans[0])), int(0.99 / spans[0])])
#     plt.xticks(np.arange(len(spans)), np.round(spans, which_decimal_s).astype(int))
#     plt.yticks(np.arange(len(centers)), np.round(centers, which_decimal_c))
#     plt.xlabel('Span')
#     plt.ylabel('Center')
#     plt.colorbar()
#     plt.title(
#         f'{name[0].upper() + name[1:]} Accuracy(%). Max:{(matrix.max() * 100).astype(int)}%@({np.round(centers[max[0]], which_decimal_c)},{np.round(spans[max[1]], which_decimal_s)})')
#     plt.tight_layout()
#     plt.savefig(os.path.join(folder_fig,
#                              f'{name}_accuracy_c{centers[0]}_{centers[-1]}_{np.round(centers[1] - centers[0], 3)}_s{spans[0]}_{spans[-1]}_{np.round(spans[1] - spans[0], 3)}.pdf'))
#     return centers[max[0]], spans[max[1]], matrix.max() * 100

def main(args):
    # Device: ----------------------------------------------------------------------------------------------------------
    device = torch.device(args.device)
    if device == torch.device("cuda") and not (torch.cuda.is_available()):
        device == torch.device('cpu')
        print('Could not find CUDA. Running on CPU')
    print(f'Running on {device}')

    if device == torch.device("cuda"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.empty_cache()
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Folders: ---------------------------------------------------------------------------------------------------------
    folder = Path('dataset_analysis')
    folder_run = Path(os.path.join(folder, 'MNIST'))
    folder_fig = folder_run.joinpath('fig')
    folder_run.mkdir(parents=True, exist_ok=True)
    folder_fig.mkdir(parents=True, exist_ok=True)
    folder_data = folder_run.joinpath('data')
    folder_data.mkdir(parents=True, exist_ok=True)

    # Dataseet: --------------------------------------------------------------------------------------------------------
    generator = set_random_seed(args.seed, add_generator=True, device=device)

    if args.n_samples_train == -1:
        # use all training samples from MNIST
        trainset = MNIST(root='data', train=True, download=False)
        n_samples_train = len(trainset)
    else:
        n_samples_train = args.n_samples_train
    if args.n_samples_test == -1:
        testset = MNIST(root='data', train=False, download=False)
        n_samples_test = len(testset)
    else:
        n_samples_test = args.n_samples_test

    path_to_dataset = Path(args.path_to_dataset)
    path_to_dataset.mkdir(parents=True, exist_ok=True)
    dict_dataset = {}

    df = {'center': [], 'span': [], 'accuracy': []}
    list_datasets = os.listdir(path_to_dataset.joinpath('MNIST', args.data_type))
    sweep = tqdm.tqdm(total=len(list_datasets), desc=f"{args.data_type} sweeping", position=0, leave=True)
    for dataset_name in list_datasets:
        filename_dataset = path_to_dataset.joinpath('MNIST',
                                                    args.data_type,
                                                    dataset_name)
        train_dataset = MNISTDataset_features(h5py.File(filename_dataset.joinpath('train.h5'), 'r'), device=device)
        test_dataset = MNISTDataset_features(h5py.File(filename_dataset.joinpath('test.h5'), 'r'), device=device)

        dict_dataset['train_loader'] = DataLoader(train_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  generator=generator)
        dict_dataset['test_loader'] = DataLoader(test_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 generator=generator)

        acc = train_classifier(dict_dataset['train_loader'], dict_dataset['test_loader'], args.n_epochs,
                               train_dataset.sample_size, device)

        df['center'].extend([train_dataset.center] * args.batch_size)
        df['span'].extend([train_dataset.span] * args.batch_size)
        df['accuracy'].extend(acc)

        sweep.update()

    with open(path_to_dataset.joinpath('MNIST', f'{args.data_type}.json'), 'w') as f:
        # write the dictionary to the file in JSON format
        json.dump(df, f)

    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('dataloader')

    parser.add_argument('--seed',
                        type=int,
                        default=10)
    parser.add_argument('--n_samples_train',
                        type=int,
                        default=20000)
    parser.add_argument('--n_samples_test',
                        type=int,
                        default=10000)
    parser.add_argument('--data_type',
                        type=str,
                        default='frequency',
                        choices=['current', 'frequency', 'amplitude', 'slope'])
    parser.add_argument('--batch_size',
                        type=int,
                        default=128)
    parser.add_argument('--n_epochs',
                        type=int,
                        default=100)
    parser.add_argument('--device',
                        type=str,
                        choices=['cpu', 'cuda'],
                        default='cpu')
    parser.add_argument('--path_to_dataset',
                        type=str,
                        default='/media/nicoletta/bics/Nicoletta/tactile_encoding/')
    args = parser.parse_args()

    # Run:
    main(args)
