import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import models
import argparse

def train(model, model_name, device, dataloader_train, dataloader_val, image_size, criterion, optimizer, epochs, save_model=False):
    """
    Train a model.
    Args:
        model: model class
        is_cnn: if the model is CNN or not
        device: device to train the model
        datalaoder_train: data loading module for training dataset
        dataloader_val: data loading module for validation dataset
        criterion: criterion to compute loss
        optimizer: optimization algorithm to train the model
        epochs: the number of epochs to train the model
    """
    
    for epoch in range(epochs):
        loss_train_total = 0
        loss_valid_total = 0
        train_correct = 0
        train_count = 0
        valid_correct = 0
        valid_count = 0

        model.to(device)
        # set the model with train mode
        model.train()
        for x, labels in dataloader_train:

            x = x.to(device)
            labels = labels.to(device)

            # reshape input tensors to (batch_size, image_size) 
            if model_name != "cnn":
                x = x.view(-1, image_size)

            y_hat = model.forward(x)

            loss_train = criterion(y_hat, labels)
            
            # back prop
            optimizer.zero_grad()
            loss_train.backward()

            # update weights
            optimizer.step()

            pred = torch.argmax(y_hat, dim=1)

            loss_train_total += loss_train.item()
            train_correct += torch.sum(pred == labels)
            train_count += 1

        loss_train_total /= train_count

        model.eval()
        for x, labels in dataloader_val:

            # put tensor on GPU
            x = x.to(device)
            labels = labels.to(device)
            
            # reshape input tensors to (batch_size, image_size)
            if model_name != "cnn":
                x = x.view(-1, image_size)

            # forward prop
            y_hat = model.forward(x)

            # compute loss
            loss_val = criterion(y_hat, labels)

            # output of the model
            pred = torch.argmax(y_hat, dim=1)

            loss_valid_total += loss_val
            valid_correct += torch.sum(pred == labels)
            valid_count += 1

        loss_valid_total /= valid_count

        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.4f}%], Valid [Loss: {:.3f}, Accuracy: {:.4f}%]'.format(
            epoch + 1,
            loss_train_total,
            100. * train_correct / len(dataloader_train.dataset),
            loss_valid_total,
            100. * valid_correct / len(dataloader_val.dataset))
        )

        if save_model:
            torch.save(model, f"outputs/mnist_{model_name}.pt")



if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logistic regression", "mlp", "cnn"], help="model for the classification", required=True)
    parser.add_argument("--hidden_units", default=[256, 128, 64], nargs=3, type=int, help="the number of units for hidden layers")
    parser.add_argument("--epochs", default=50, type=int, help="the number of epochs for training", required=True)
    parser.add_argument("--batch_size", default=64, type=int, help="batch size for training", required=True)
    parser.add_argument("--optimizer", choices=["adam", "rmsprop", "sgd"], help="optimization algorithm", required=True)
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="learning rate", required=True)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--model_save", type=bool, default=False, required=True)
    args = parser.parse_args()

    # transform image data to tensor compatible with pytorch
    to_tensor = transforms.ToTensor()
    # normalize image data
    normalize = transforms.Normalize((0.5, ), (0.5, ))
    # compose as a transform class 
    transform = transforms.Compose([to_tensor, normalize])

    # load train and test dataset
    train_dataset = datasets.FashionMNIST(
            root="/home/tatsu/dl_portfolio/mnist-fashion/dataset", train=True, transform=transform, download=True
    )

    # define batch size, validation dataset size
    batch_size = args.batch_size
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

    in_dims = 784
    out_dims = 10
    model_name = args.model

    if model_name == "mlp":
        hidden_units = args.hidden_units
        model = models.MLP(in_dims=in_dims, out_dims=out_dims, hidden_units=hidden_units)
    elif model_name == "logistic regression":
        model = models.LogisticRegressor(in_dims, out_dims)
    elif model_name == "cnn":
        model = models.CNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 28*28
    criterion = nn.CrossEntropyLoss()
    lr = args.learning_rate
    weight_decay = args.weight_decay

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif args.optimizer == "sgd":
        optimizer == optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    epochs = args.epochs
    model_save = args.model_save

    train(model, model_name, device, dataloader_train, dataloader_val, image_size, criterion, optimizer, epochs, model_save)
