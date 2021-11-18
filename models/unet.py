import torch.nn.functional as F
import torch.nn as nn
import torch

class block(nn.Module):
    def __init__(self, in_channel, out_channel, encoder=True):
        super(block, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.encoder = encoder # 앞부분의 downsampling하는 것인지

    def forward(self, x):
        if self.encoder == True:
            x = self.pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_channel, out_channel): # 256x256x8 image(random crop applied)
        super(Unet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, 1, 1), # kernel, stride, padding
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.enc2 = block(32, 64)
        self.up1 = nn.ConvTranspose2d(64,32,4,2,1)
        self.dec4 = block(64, 32, encoder=False)

        self.enc3 = block(64, 128)
        self.up2 = nn.ConvTranspose2d(128,64,4,2,1)
        self.dec3 = block(128, 64, encoder=False)

        self.enc4 = block(128, 256)
        self.up3 = nn.ConvTranspose2d(256,128,4,2,1)
        self.dec2 = block(256, 128, encoder=False)

        self.enc5 = block(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 4, 2, 1)

        self.dec1 = block(512, 256, encoder=False)
        self.conv = nn.Conv2d(32,2,1,1) # final layer

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        x6 = torch.cat((self.up4(x5), x4), 1) # channel wise concatenation(B, C, H, W)
        x7 = torch.cat((self.up3(self.dec1(x6)), x3), 1)
        x8 = torch.cat((self.up2(self.dec2(x7)), x2), 1)
        x9 = torch.cat((self.up1(self.dec3(x8)), x1), 1)
        
        x9 = self.dec4(x9)
        x9 = self.conv(x9)
        return x9