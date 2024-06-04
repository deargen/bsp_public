import torch.nn as nn 
import torch 




class BottleneckResnet3dUnit(nn.Module):
    def __init__(self, last_channel, channel, stride):
        super().__init__()
        self.in_channel = last_channel 
        self.out_channel = channel 
        assert channel % 4 == 0
        self.bottleneck_channel = channel // 4
        self.stride = stride
        
        if self.in_channel == self.out_channel:
            if stride == 1:
                self.to_shortcut = nn.Identity()
            else:
                self.to_shortcut = nn.MaxPool3d(kernel_size=self.stride, stride=self.stride, padding=0)
        else:
            self.to_shortcut = nn.Conv3d(self.in_channel, self.out_channel, kernel_size=1, stride=self.stride, padding=0, bias=False) #bias:False since batchnorm has affine==True 
            nn.init.xavier_uniform_(self.to_shortcut.weight)
        
        self.conv_act_fn = nn.ReLU()
        self.conv1 = nn.Conv3d(self.in_channel, self.bottleneck_channel, kernel_size=1, stride=1, padding=0, bias=False) #bias:False since there's batchnorm 
        self.bn1 = nn.BatchNorm3d(self.bottleneck_channel, affine=True, momentum=0.01)
        self.conv2 = nn.Conv3d(self.bottleneck_channel, self.bottleneck_channel, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(self.bottleneck_channel, affine=True, momentum=0.01)
        self.conv3 = nn.Conv3d(self.bottleneck_channel, self.out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(self.out_channel, affine=True, momentum=0.01)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        
        self.final_act_fn = nn.ReLU()
        
    
    def forward(self, x):
        shortcut = self.to_shortcut(x)
        
        x = self.conv_act_fn(self.bn1(self.conv1(x)))
        x = self.conv_act_fn(self.bn2(self.conv2(x)))
        x = self.conv_act_fn(self.bn3(self.conv3(x)))
        
        return self.final_act_fn(shortcut + x)

class BottleneckResnet3dBlock(nn.Module):
    def __init__(self, last_channel, channel, stride, unit):
        super().__init__() 
        assert channel % 4 == 0 
        if unit == 1:
            triples = [(last_channel, channel, stride)]
        else:
            assert unit >= 2
            triples = [(last_channel, channel, 1)] + [(channel, channel, 1)] * (unit - 2) + [(channel, channel, stride)]
        self.units = nn.ModuleList([BottleneckResnet3dUnit(channel1, channel2, stride0) for (channel1, channel2, stride0) in triples])
    def forward(self, x):
        for unit in self.units:
            x = unit(x)
        return x 

class BottleneckResnet3d(nn.Module):
    def __init__(self, last_channel, channels, strides, units):
        super().__init__()
        last_channels = [last_channel] + channels[:-1]
        self.blocks = nn.ModuleList([BottleneckResnet3dBlock(_last_channel, channel, stride, unit) for (_last_channel, channel, stride, unit) in zip(last_channels, channels, strides, units)])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return torch.mean(x, dim=(-3, -2, -1))

    
        
if __name__ == '__main__':
    model = BottleneckResnet3d(18, [64, 128, 256, 512], [2, 2, 2, 1], [2, 2, 2, 2])
    x = torch.rand((8, 18, 16, 16, 16))
    y = model(x)
    print('x.shape:', tuple(x.shape))
    print('y.shape:', tuple(y.shape))
    