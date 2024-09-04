import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

try:
    from .nafnet_utils.arch_util import LayerNorm2d
    from .nafnet_utils.arch_model import SimpleGate
except:
    from nafnet_utils.arch_util import LayerNorm2d
    from nafnet_utils.arch_model import SimpleGate


class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''
    def __init__(self, c, DW_Expand, dilation = 1, extra_depth_wise = False):
        super().__init__()
        self.dw_channel = DW_Expand * c 
        self.branch = nn.Sequential(
                       nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity(), #optional extra dw
                       nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1),
                       nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation, stride=1, groups=self.dw_channel,
                                            bias=True, dilation = dilation) # the dconv
        )
    def forward(self, input):
        return self.branch(input)
    
class EBlock(nn.Module):
    '''
    Change this block using Branch
    '''
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation = dilation, extra_depth_wise=extra_depth_wise))
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):

        y = inp
        x = self.norm1(inp)
        z = 0
        for branch in self.branches:
            z += branch(x)
        
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        #second step
        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg2(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        return y + x * self.gamma

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    img_channel = 3
    width = 32

    enc_blks = [1, 2, 3]
    middle_blk_num = 3
    dec_blks = [3, 1, 1]
    dilations = [1, 4, 9]
    extra_depth_wise = False
    
    # net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
    #                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    net  = EBlock(c = img_channel, 
                            dilations = dilations,
                            extra_depth_wise=extra_depth_wise)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)

    # print('Values of EBlock:')
    # print(macs, params)

    channels = 128
    resol = 32
    ksize = 5

    net = FAC(channels=channels, ksize=ksize)
    inp_shape = (channels, resol, resol)
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)
    print('Values of FAC:')
    print(macs, params)

