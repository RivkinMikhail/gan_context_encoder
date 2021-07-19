import torch
import torch.nn as nn


class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            nn.Conv2d(opt.nc,opt.imageSize,4,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(opt.imageSize,opt.imageSize,4,2,1, bias=False),
            nn.BatchNorm2d(opt.imageSize),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(opt.imageSize,opt.imageSize*2,4,2,1, bias=False),
            nn.BatchNorm2d(opt.imageSize*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(opt.imageSize*2,opt.imageSize*4,4,2,1, bias=False),
            nn.BatchNorm2d(opt.imageSize*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.imageSize*4,opt.imageSize*8,4,2,1, bias=False),
            nn.BatchNorm2d(opt.imageSize*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.imageSize*8,opt.nBottleneck,4, bias=False),

            nn.BatchNorm2d(opt.nBottleneck),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(opt.nBottleneck, opt.cropSize * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.cropSize * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.cropSize * 8, opt.cropSize * 4, 4, 2, 0, bias=False),
            nn.BatchNorm2d(opt.cropSize * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.cropSize * 4, opt.cropSize * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(opt.cropSize * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.cropSize * 2, opt.cropSize, 4, 1, 1, bias=False),
            nn.BatchNorm2d(opt.cropSize),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.cropSize, opt.nc, 4, 1, 0, bias=False),
            nn.Tanh()

        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netlocalD(nn.Module):
    def __init__(self, opt):
        super(_netlocalD, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(

            nn.Conv2d(opt.nc,  opt.cropSize, 3, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.cropSize,  opt.cropSize * 2, 3, 3, 1, bias=False),
            nn.BatchNorm2d(opt.cropSize * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.cropSize * 2,  opt.cropSize * 4, 3, 3, 1, bias=False),
            nn.BatchNorm2d(opt.cropSize * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.cropSize * 4,  opt.cropSize * 8, 3, 3, 1, bias=False),
            nn.BatchNorm2d(opt.cropSize * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.cropSize * 8, 1, 1, 3, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1)

