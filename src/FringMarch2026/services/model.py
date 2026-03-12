import torch.nn as nn
import torch

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        def block(in_c,out_c):
            return nn.Sequential(
                nn.Conv2d(in_c,out_c,3,padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c,out_c,3,padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.enc1 = block(1,64)
        self.pool = nn.MaxPool2d(2)

        self.enc2 = block(64,128)
        self.enc3 = block(128,256)

        self.mid = block(256,512)

        self.up3 = nn.ConvTranspose2d(512,256,2,2)
        self.dec3 = block(512,256)

        self.up2 = nn.ConvTranspose2d(256,128,2,2)
        self.dec2 = block(256,128)

        self.up1 = nn.ConvTranspose2d(128,64,2,2)
        self.dec1 = block(128,64)

        self.upscale = nn.ConvTranspose2d(64,32,2,2)

        self.out = nn.Conv2d(32,1,1)

    def forward(self,x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.mid(self.pool(e3))

        d3 = self.up3(m)
        d3 = self.dec3(torch.cat([d3,e3],1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2,e2],1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1,e1],1))

        up = self.upscale(d1)

        return self.out(up)