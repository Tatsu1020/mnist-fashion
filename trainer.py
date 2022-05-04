import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.logistic_regression import LogisticRegressor


def train(model, device, dataloader_train, dataloader_val, criterion, optimizer, epochs):
    """
    """

    for epoch in range(epochs):
        losses_train = []
        losses_valid = []
        train_num = 0
        train_true_num = 0
        valid_num = 0
        valid_true_num = 0

        model.to(device)
        # set the model with train mode
        model.train()
        for x, label in dataloader_train:
            true = label.tolist()
            one_hot = torch.eye(10)[label]

            x = x.to(device)
            one_hot = one_hot.to(device)

            y_hat = torch.squeeze(model.forward(x))

            loss = criterion(y_hat, one_hot)

            pred = torch.argmax(y_hat, dim=1)

            losses_train.append(loss.tolist())

            acc = torch.where(label - pred.to("cpu") == 0, torch.ones_like(label), torch.zeros_like(label))
            train_num += acc.size()[0]
            train_true_num += acc.sum().item


        model.eval()
        for x, label in dataloader_val:
            # WRITE ME
            true = label.tolist()
            one_hot = torch.eye(10)[label]

            # put tensor on GPU
            x = x.to(device)
            one_hot = one_hot.to(device)

            # forward prop
            y_hat = torch.squeeze(model.forward(x))

            # compute loss
            loss = criterion(y_hat, one_hot)

            # output of the model
            pred = torch.argmax(y_hat, dim=1)

            losses_valid.append(loss.tolist())

            acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(label), torch.zeros_like(label))
            valid_num += acc.size()[0]
            valid_true_num += acc.sum().item()

        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
            epoch,
            np.mean(losses_train),
            train_true_num/train_num,
            np.mean(losses_valid),
            valid_true_num/valid_num
        ))



if __name__ == "__main__":

    # transform image data to tensor compatible with pytorch
    to_tensor = transforms.ToTensor()

    # load train and test dataset
    train_dataset = datasets.FashionMNIST(
            root="/home/tatsu/dl_portfolio/mnist-fashion/dataset", train=True, transform=to_tensor, download=True
    )
    test_data = datasets.FashionMNIST(
            root="/home/tatsu/dl_portfolio/mnist-fashion/dataset", train=False, transform=to_tensor, download=True
    )
    
    train_dataset = train_dataset.reshape(-1, 784).astype("float32") / 255

    # define batch size, validation dataset size
    batch_size = 64
    val_size = 10000
    train_size = len(train_dataset) - val_size

    # split the dataset into train and val data
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # define dataloader for train, val, and test data
    dataloader_train = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
    )

    dataloader_val = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=True
    )

    dataloader_test = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=True
    )

    model = LogisticRegressor(784, 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs = 50

    train(model, device, dataloader_train, dataloader_val, criterion, optimizer, epochs)
