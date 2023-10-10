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


def train_classifier(train_dl, test_dl, n_epochs, device):
    examples = enumerate(train_dl)
    batch_idx, (example_data, example_targets) = next(examples)

    classifier = nn.Sequential(nn.Linear(example_data.shape[1], train_dl.dataset.n_classes, )).to(device)
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


def main(args):
    print(f'Running with data type on {args.data_type}')

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

        acc = train_classifier(dict_dataset['train_loader'], dict_dataset['test_loader'], args.n_epochs, device)

        df['center'].extend([train_dataset.center] * len(acc))
        df['span'].extend([train_dataset.span] * len(acc))
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
    parser.add_argument('--data_type',
                        type=str,
                        default='amplitude',
                        choices=['frequency', 'amplitude', 'slope'])
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
                        default='./dataset/')
    args = parser.parse_args()

    # Run:
    main(args)
