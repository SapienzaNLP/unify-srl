import torch
import torch.nn as nn


class StateEncoder(nn.Module):

    def __init__(self, input_size, state_size, num_layers, activation, dropout_rate):
        super(StateEncoder, self).__init__()
        _linears = []
        for i in range(num_layers):
            if i == 0:
                _linears.append(nn.Linear(input_size, state_size))
            else:
                _linears.append(nn.Linear(state_size, state_size))

        self.linears = nn.ModuleList(_linears)

        if activation == 'identity':
            self.activation = nn.Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = Swish()

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
