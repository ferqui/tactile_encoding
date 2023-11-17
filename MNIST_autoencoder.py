import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision as tv
from tqdm import tqdm
from utils import set_random_seed
from utils_dataset import Autoencoder_linear
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

    trainset = MNIST(root='data', train=True, download=True)
    list_transforms = [transforms.ToTensor(), transforms.Normalize((trainset.data.float().mean() / 255,),
                                                                   (trainset.data.float().std() / 255,))]

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
    dataloader = train_loader

    # Encoder model:
    #model = Autoencoder(args.encoding_dim)
    model = Autoencoder_linear(args.encoding_dim,
                               input_dim=trainset.data.shape[1]**2,
                               output_dim=args.output_dim)

    # Loss function
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    list_loss = []
    # Training:
    for epoch in range(1, args.n_epochs + 1):
        # monitor training loss
        train_loss = 0.0

        for data in tqdm(dataloader):
            # _ stands in for labels, here
            images, target = data

            # flatten images
            # print(images.shape)
            # print(target.shape)
            images = images.view(images.size(0), -1).float()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images.float())
            # calculate the loss
            # print(outputs.float().shape)
            # print(images.shape)
            # print(outputs)

            loss = loss_fn(outputs, target)

            #loss = criterion(outputs.float(), images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)
            list_loss.append(loss.item())

        # print avg training statistics
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss/len(dataloader)
        ))

    plt.figure()
    plt.plot(list_loss)
    plt.show()
    # Plot output:
    # obtain one batch of test images
    # obtain one batch of test images
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    images_flatten = images.view(images.size(0), -1)
    assert all(images_flatten[0] == images[0].flatten())

    # get sample outputs
    output = model(images_flatten.float())
    encs = model.encoder(images_flatten.float()).clone().detach()
    print(encs.shape)
    encs = torch.reshape(encs, (args.batch_size, 3, 2))
    # prep images for display
    images = images.numpy()

    # output is resized into a batch of images
    #output = output.view(args.batch_size, 1, 28, 28)
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

    #assert all(images_flatten[0] == torch.tensor(images)[0].flatten())

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
                        default=6)
    parser.add_argument('--output_dim',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        type=int,
                        default=32)
    parser.add_argument('--n_epochs',
                        type=int,
                        default=20)
    args = parser.parse_args()

    # Run test:
    main(args)