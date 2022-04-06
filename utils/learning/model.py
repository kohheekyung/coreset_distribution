import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt

class Distribution_Model(nn.Module):
    """
    Default LinearNet which has 3 fc layers
    """
    def __init__(self, args, input_size, output_size):
        # input_size : ~1536 * 8, output_size : ~512
        super().__init__()
        self.l1 = nn.Linear(input_size, 2048)
        self.l2 = nn.Linear(2048, 1024)
        self.l3 = nn.Linear(1024, output_size)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)

        return x