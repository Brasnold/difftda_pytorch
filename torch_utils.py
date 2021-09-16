import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data


class RipsFiltrationLayer(nn.Module):

    def __init__(self):
        super(RipsFiltrationLayer, self).__init__()

        self.linear = nn.Linear(n_prods * 2, n_prods + 1)

    def forward(self, state):
        x = self.linear(state)

        return x