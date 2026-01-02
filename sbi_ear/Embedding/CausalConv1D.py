import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Conv1d 

class CausalConv1d(nn.Module):
    """
    A causal 1D convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride, A=False, **kwargs):
        super(CausalConv1d, self).__init__()

        # attributes:
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.A = A
        
        self.padding = (kernel_size - 1) * dilation + A * 1

        # module:
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels,
                                      kernel_size, stride,
                                      padding=0,
                                      dilation=dilation,
                                      **kwargs)


    def forward(self, x):
        # x = torch.nn.functional.pad(x, (self.padding, 0))
        conv1d_out = self.conv1d(x)

        if self.A:
            return conv1d_out[:, :, : -1]
        else:
            return conv1d_out
        

    

class CausalConvLayers(nn.Module):

    def __init__(self, in_channels, out_channels, MM, stride, kernel_size):
        super(CausalConvLayers, self).__init__()

        self.layers= nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=32, dilation=1, kernel_size=64, A=True, bias = True, stride = 2), #
            nn.LeakyReLU(), 
            CausalConv1d(in_channels=32, out_channels=32, dilation=2, kernel_size=64, A=False, bias = True, stride = 2), 
            nn.LeakyReLU(), 
            CausalConv1d(in_channels=32, out_channels=32, dilation=4, kernel_size=64, A=False, bias = True, stride = 2), 
            nn.LeakyReLU(), 
            CausalConv1d(in_channels=32, out_channels=out_channels, dilation=8, kernel_size=64, A=False, bias = True,  stride = 2), 
            nn.LeakyReLU(),  
            )
        self.layers8 = nn.Sequential(
                                    CausalConv1d(in_channels=1, out_channels=32, dilation= 1, kernel_size=64, A=True, bias = True, stride = 1), #
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 2, kernel_size=64, A=False, bias = True, stride = 1), 
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 2, kernel_size=64, A=False, bias = True, stride = 1), 
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 4, kernel_size=64, A=False, bias = True,  stride = 1), 
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 4, kernel_size=64, A=True, bias = True, stride = 1), #
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 16, kernel_size=64, A=False, bias = True, stride = 1), 
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 16, kernel_size=64, A=False, bias = True, stride = 1), 
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=1, dilation= 32, kernel_size=64, A=False, bias = True,  stride = 1), 
                                    nn.LeakyReLU(),  
                                    )  #1291
        
        self.layers12 = nn.Sequential(
                                    CausalConv1d(in_channels=1, out_channels=32, dilation= 1, kernel_size=32, A=True, bias = True, stride = 1), #
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 2, kernel_size=32, A=False, bias = True, stride = 1), 
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 2, kernel_size=32, A=False, bias = True, stride = 1), 
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 4, kernel_size=32, A=False, bias = True,  stride = 1), 
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 4, kernel_size=32, A=True, bias = True, stride = 1), #
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 8, kernel_size=32, A=False, bias = True, stride = 1), 
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 8, kernel_size=32, A=False, bias = True, stride = 1), 
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 16, kernel_size=32, A=False, bias = True,  stride = 1), 
                                    nn.LeakyReLU(),     
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 16, kernel_size=32, A=False, bias = True,  stride = 1), 
                                    nn.LeakyReLU(),        
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 32, kernel_size=32, A=False, bias = True, stride = 1), 
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=32, dilation= 32, kernel_size=32, A=False, bias = True,  stride = 1), 
                                    nn.LeakyReLU(), 
                                    CausalConv1d(in_channels=32, out_channels=1, dilation= 32, kernel_size=32, A=False, bias = True,  stride = 1), 
                                    nn.LeakyReLU(), 

                                    ) #1275
        self.flatten = nn.Flatten()

    def forward(self,x):
        # print(x.size())
        x = self.layers12(x.unsqueeze(1))
        x = self.flatten(x)
        return x

par