import torch.nn as nn
import torch.nn.functional as F


# pixel-wise softmax
# use building mask as training data instead of signed distance transform
# changed parameters from baseline

class Block(nn.Module):
    def __init__(self, in_channel, mid, out_channel, first=False, last=False): #256
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(mid)
        self.transConv1 = nn.ConvTranspose2d(mid, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.dp = nn.Dropout(p=0.3, inplace=True)
        self.first = first
        self.last = last

    def forward(self, x):
        if not self.first:
          x = self.dp(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.transConv1(x)
        x = self.bn2(x)
        x = self.relu(x)

        if not self.last:
            x = self.pool1(x)
        return x

class FCnet(nn.Module):  # input : 650x650x8 consider crop later
    def __init__(self, inp):
        super(FCnet, self).__init__()
        self.block1 = Block(inp, 16, 32)
        self.block2 = Block(32, 32, 16)
        self.block3 = Block(16, 16, 16, last=True)
        self.transconv = nn.ConvTranspose2d(16, 2, 3, padding=1) # should outputs 2 dim
        

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)   
        x = self.transconv(x)

        return x

