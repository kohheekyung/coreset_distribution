import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt

class Distribution_Model(nn.Module):
    """
    Default LinearNet which has 3 fc layers
    """
    def __init__(self, args, input_size, output_size):
        # input_size : ~1536 * 8, output_size : ~256
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, 1024)
        self.l2 = torch.nn.Linear(1024, 512)
        self.l3 = torch.nn.Linear(512, output_size)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)

        return x