import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import set_random_seed
from utils_dataset import load_MNIST, Autoencoder
import argparse
import torch.nn as nn
import seaborn as sns
sns.set_style('white')
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from torchvision.datasets import MNIST

def main(args):

    generator = set_random_seed(args.seed, add_generator=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: num_workers > 0 and pin_memory True does not work on pytorch 1.12
    # try pytorch 1.13 with CUDA > 1.3
    kwargs = {'num_workers': 4, 'pin_memory': True}

    list_transforms = [transforms.PILToTensor()]

    # Train:
    trainset = MNIST(root='data', train=True, download=True,
                     transform=transforms.Compose(list_transforms))
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              generator=generator,
                              **kwargs)
    print(f'N samples training: {len(trainset.data)}')

    # Test:
    testset = MNIST(root='data', train=False, download=True,
                    transform=transforms.Compose(list_transforms))
    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             generator=generator,
                             **kwargs)
    print(f'N samples test: {len(testset.data)}')


    # Encoder model:
    model = Autoencoder(args.encoding_dim)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training:
    for epoch in range(1, args.n_epochs + 1):
        # monitor training loss
        train_loss = 0.0

        for data in tqdm(train_loader):
            # _ stands in for labels, here
            images, _ = data

            # flatten images
            images = images.view(images.size(0), -1).float()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images.float())
            # calculate the loss
            loss = criterion(outputs.float(), images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)

        # print avg training statistics
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
        ))

    # Plot output:
    # obtain one batch of test images
    # obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    images_flatten = images.view(images.size(0), -1)
    assert all(images_flatten[0] == images[0].flatten())

    # get sample outputs
    output = model(images_flatten.float())
    encs = model.encoder(images_flatten.float()).clone().detach()
    print(encs.shape)
    encs = torch.reshape(encs, (args.batch_size, 6, 4))
    # prep images for display
    images = images.numpy()

    # output is resized into a batch of images
    output = output.view(args.batch_size, 1, 28, 28)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(25, 4))

    # input images on top row, reconstructions on bottom
    for images_, row in zip([images, encs, output], axes):
        for img, ax in zip(images_, row):
            sns.heatmap(np.squeeze(img), ax=ax, cmap='Greys')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    assert all(images_flatten[0] == torch.tensor(images)[0].flatten())

    fig.savefig('./data/MNIST_autoencoder.pdf', format='pdf')
    fig.savefig('./data/MNIST_autoencoder.png', format='png', dpi=300)

    # Store model:
    torch.save(model.state_dict(), f'./data/784MNIST_2_{args.encoding_dim}MNIST.pt')
    model.state_dict()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train')

    parser.add_argument('--seed',
                        type=int,
                        default=42)
    parser.add_argument('--encoding_dim',
                        type=int,
                        default=24)
    parser.add_argument('--batch_size',
                        type=int,
                        default=200)
    parser.add_argument('--n_epochs',
                        type=int,
                        default=50)
    args = parser.parse_args()

    # Run test:
    main(args)