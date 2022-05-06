import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import models
import argparse

def test(model, model_name, device, dataloader_test, image_size, criterion):
    """
    Train a model.
    Args:
        model: model class
        model_name: the model architecture
        device: device to train the model
        datalaoder_test: data loading module for test dataset
        criterion: criterion to compute loss
    """
    
    model.eval()
    test_loss = 0
    correct = 0
    count = 0

    with torch.no_grad():
        for x, labels in dataloader_test:
            x, labels = x.to(device), labels.to(device)
            
            if model_name != "cnn":
                x = x.view(-1, image_size)

            y_hat = model(x)
            loss = criterion(y_hat, labels).item()
            test_loss += loss
            pred = torch.argmax(y_hat, dim=1)
            correct += torch.sum(pred == labels)
            count += 1

    test_loss /= count

    print("\nTest [Loss: {:.3f}, Accuracy: {}/{} ({:.3f}%)]\n".format(
        test_loss, 
        correct,
        len(dataloader_test.dataset),
        100. * correct / len(dataloader_test.dataset))
    )


if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to the file containing a trained model", required=True)
    parser.add_argument("--model_name", choices=["cnn", "mlp", "logistic_regression"], required=True)
    parser.add_argument("--batch_size", default=64, type=int, help="batch size for training", required=True)
    args = parser.parse_args()

    # transform image data to tensor compatible with pytorch
    to_tensor = transforms.ToTensor()
    # normalize image data
    normalize = transforms.Normalize((0.5, ), (0.5, ))
    # compose as a transform class 
    transform = transforms.Compose([to_tensor, normalize])
    
    # instantiate test dataset
    test_data = datasets.FashionMNIST(
            root="/home/tatsu/dl_portfolio/mnist-fashion/dataset", train=False, transform=transform, download=True
    )

    # define batch size, validation dataset size
    batch_size = args.batch_size
    
    # create a dataloader
    dataloader_test = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=True
    )
    
    # define all necessary aqrguments
    image_size = 28*28
    model_name = args.model_name
    model_path = args.model_path 
    model = torch.load(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # evaluate the model
    test(model, model_name, device, dataloader_test, image_size, criterion)
