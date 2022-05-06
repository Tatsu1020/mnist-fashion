import torch
import torch.nn as nn

class LogisticRegressor(nn.Module):
    """
    Losgistic regression with pytorch.
    """

    def __init__(self, in_dims, out_dims):
        super(LogisticRegressor, self).__init__()
        self.linear = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        x = nn.functional.softmax(self.linear(x), dim=1)
        return x

class MLP(nn.Module):
    """
    Multi perceptron with pytorch.
    """

    def __init__(self, in_dims, out_dims, hidden_units):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dims, hidden_units[0])
        self.linear2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.linear3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.linear4 = nn.Linear(hidden_units[2], out_dims)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.linear1(x))
        x = nn.functional.leaky_relu(self.linear2(x))
        x = nn.functional.leaky_relu(self.linear3(x))
        x = nn.functional.softmax(self.linear4(x), dim=1)
        return x

class CNN(nn.Module):
    """
    Three layers convolutional neural netowrks with 2 fully connected layers with pytorch.
    """

    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.linear_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*64, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.linear_projection(x)
        return x

