import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.logistic_regression import LogisticRegressor


def train(model, device, dataloader_train, dataloader_val, image_size, criterion, optimizer, epochs):
    """
    Train a model.
    Args:
        model: model class
        device: device to train the model
        datalaoder_train: data loading module for training dataset
        dataloader_val: data loading module for validation dataset
        criterion: criterion to compute loss
        optimizer: optimization algorithm to train the model
        epochs: the number of epochs to train the model
    """

    for epoch in range(epochs):
        losses_train = []
        losses_valid = []
        train_num = 0
        train_acc = 0
        valid_num = 0
        valid_acc = 0

        model.to(device)
        # set the model with train mode
        model.train()
        for x, labels in dataloader_train:

            x = x.to(device)
            labels = labels.to(device)

            # reshape input tensors to (batch_size, image_size) 
            x = x.view(-1, image_size)

            y_hat = model.forward(x)

            loss = criterion(y_hat, labels)
            
            # back prop
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            pred = torch.argmax(y_hat, dim=1)

            losses_train.append(loss.tolist())

            train_num += x.size()[0]
            train_acc += torch.sum(pred == labels).item()

        model.eval()
        for x, labels in dataloader_val:

            # put tensor on GPU
            x = x.to(device)
            labels = labels.to(device)
            
            # reshape input tensors to (batch_size, image_size)
            x = x.view(-1, image_size)

            # forward prop
            y_hat = model.forward(x)

            # compute loss
            loss = criterion(y_hat, labels)

            # output of the model
            pred = torch.argmax(y_hat, dim=1)

            losses_valid.append(loss.tolist())

            valid_num += x.size()[0]
            valid_acc += torch.sum(pred == labels)


        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
            epoch,
            np.mean(losses_train),
            train_acc/train_num,
            np.mean(losses_valid),
            valid_acc/valid_num
        ))



if __name__ == "__main__":

    # transform image data to tensor compatible with pytorch
    to_tensor = transforms.ToTensor()
    # normalize image data
    normalize = transforms.Normalize((0.5, ), (0.5, ))
    
    transform = transforms.Compose([to_tensor, normalize])

    # load train and test dataset
    train_dataset = datasets.FashionMNIST(
            root="/home/tatsu/dl_portfolio/mnist-fashion/dataset", train=True, transform=transform, download=True
    )
    test_data = datasets.FashionMNIST(
            root="/home/tatsu/dl_portfolio/mnist-fashion/dataset", train=False, transform=transform, download=True
    )
    
    #train_dataset = train_dataset.reshape(-1, 784).astype("float32") / 255

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
    image_size = 28*28
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs = 50

    train(model, device, dataloader_train, dataloader_val, image_size, criterion, optimizer, epochs)
