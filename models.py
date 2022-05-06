import torch
import torch.nn as nn

class LogisticRegressor(nn.Module):
    """
    Losgistic regression model with pytorch.
    """

    def __init__(self, in_dims, out_dims):
        super(LogisticRegressor, self).__init__()
        self.linear = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        x = nn.functional.softmax(self.linear(x), dim=1)
        return x



