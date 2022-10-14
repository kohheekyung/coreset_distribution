import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt

class Distribution_Model(nn.Module):
    """
    Default LinearNet which has 3 fc layers
    """
    def __init__(self, args, input_size, output_size):
        # input_size : ~1024 * 24, output_size : ~2048
        super().__init__()
        num_layers = args.num_layers
        self.first_layer = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout()
        )
        self.last_layer = nn.Sequential(
            nn.Linear(2048, output_size)
        )
        self.hidden_layers = nn.ModuleList()
        for i in range(1, num_layers-1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout()
            ))
            
        """
        self.fcs = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, output_size)
        )
        """

    def forward(self, x):
        x1 = self.first_layer(x)
        for layer in self.hidden_layers :
            x1 = layer(x1)
        out = self.last_layer(x1)
        
        return out
    
    
class Refine_Model(nn.Module) :
    
    def __init__(self, in_chans, out_chans, first_channels = 32):
        super().__init__()
        self.bottom_param = first_channels * 16
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.first_block = ConvBlock(in_chans, self.bottom_param//16) 
        self.down1 = Down(self.bottom_param//16, self.bottom_param//8) 
        self.down2 = Down(self.bottom_param//8, self.bottom_param//4) 
        self.down3 = Down(self.bottom_param//4, self.bottom_param//2) 
        self.down4 = Down(self.bottom_param//2, self.bottom_param)
        self.up1 = Up(self.bottom_param, self.bottom_param//2) 
        self.up2 = Up(self.bottom_param//2, self.bottom_param//4)
        self.up3 = Up(self.bottom_param//4, self.bottom_param//8) 
        self.up4 = Up(self.bottom_param//8, self.bottom_param//16) 
        self.last_block = nn.Conv2d(self.bottom_param//16, out_chans, kernel_size = 1)

    def norm(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, torch.mean(mean, dim=1).view(b, 1, 1, 1), torch.mean(std, dim=1).view(b, 1, 1, 1)

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, input):
        #input, mean, std = self.norm(input)
        
        d1 = self.first_block(input) 
        d2 = self.down1(d1) 
        d3 = self.down2(d2) 
        d4 = self.down3(d3) 
        m0 = self.down4(d4)
        u1 = self.up1(m0, d4) 
        u2 = self.up2(u1, d3) 
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1) 
        output = self.last_block(u4)
        #output = self.unnorm(output, mean, std)
        #output = torch.sigmoid(output) # output probability (0 to 1)
        
        return output
        
        #return torch.squeeze(output, dim=1) # (N, 384, 384) if out_chans is 1


class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_chans, out_chans)
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, out_chans)

    def forward(self, x, concat_input):
        x = self.up(x)
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)
    