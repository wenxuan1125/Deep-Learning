import torch.nn as nn
from torch import Tensor

class BasicBlock(nn.Module):
    # custermize our neural network block -> inherit class "nn.Module" and implement method "forward"
    # backward will be automatically implemented when "forward" is implemented
    # layers with weights to be learned are put in constructor "__init__()" 
    # layers without weights to be learned can be put in constructor "__init__()" or "forward"
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: bool = False):
        super(BasicBlock, self).__init__()  # call constructor of nn.Module

        # torch.nn.Conv2d(in_channels of input image, out_channels produced by the convolution, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        #       groups = number of blocked connections from input channels to output channels
        #       input = (N,C,H,W) or (C,H,W) = (# of data, # of channels of input image, height of image, width of image)
        #       output = (N,C,H,W) or (C,H,W) = (# of data, # of channels of output image, height of image, width of image)
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        #       num_features = C from an expected input of size (N,C,H,W)
        #       input = (N,C,H,W) or (C,H,W)
        #       output = (N,C,H,W) or (C,H,W)
        # torch.nn.ReLU(inplace=False)
        #       inplace: 是否将得到的值计算得到的值覆盖之前的值
        #       -> reference1 https://www.cnblogs.com/wanghui-garcia/p/10642665.html
        #       -> reference2 https://zhuanlan.zhihu.com/p/350316775
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=(1, 1),bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=(1, 1),bias=False),
            nn.BatchNorm2d(out_channels),
            
        )
        self.relu = nn.ReLU(inplace=True)

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                
            )
        else: 
            self.downsample = lambda x:x
        
    def forward(self, x) -> Tensor:
        output = self.conv1(x)
        output = self.conv2(output)
        identity = self.downsample(x)
        output = self.relu(output + identity)

        return output

class BottleneckBlock(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1, downsample: bool = False):
        super(BottleneckBlock, self).__init__()  # call constructor of nn.Module

        # torch.nn.Conv2d(in_channels of input image, out_channels produced by the convolution, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        #       groups = number of blocked connections from input channels to output channels
        #       input = (N,C,H,W) or (C,H,W) = (# of data, # of channels of input image, height of image, width of image)
        #       output = (N,C,H,W) or (C,H,W) = (# of data, # of channels of output image, height of image, width of image)
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        #       num_features = C from an expected input of size (N,C,H,W)
        #       input = (N,C,H,W) or (C,H,W)
        #       output = (N,C,H,W) or (C,H,W)
        # torch.nn.ReLU(inplace=False)
        #       inplace: 是否将得到的值计算得到的值覆盖之前的值
        #       -> reference1 https://www.cnblogs.com/wanghui-garcia/p/10642665.html
        #       -> reference2 https://zhuanlan.zhihu.com/p/350316775
        # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1,1), stride=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, (1, 1), stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                
            )
        else: 
            self.downsample = lambda x:x
        
    def forward(self, x) -> Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        identity = self.downsample(x)
        out = self.relu(out + identity)

        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.name = 'resnet18'

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(64, 64, stride=1, downsample=False)
        self.layer2 = self.make_layer(64, 128, stride=2, downsample=True)
        self.layer3 = self.make_layer(128, 256, stride=2, downsample=True)
        self.layer4 = self.make_layer(256, 512, stride=2, downsample=True)

        # torch.nn.AdaptiveAvgPool2d(output_size = (H, W)):
        #       Applies a 2D adaptive average pooling over an input signal composed of several input planes.
        #       The output is of size H x W, for any input size. The number of output features is equal to the number of input planes.
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 5)
        )


    def make_layer(self, in_channels: int, out_channels: int, stride: int = 1, downsample: bool = False):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample),
            BasicBlock(out_channels, out_channels)
        )

    def forward(self, x) -> Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out)

        return out

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.name = 'resnet50'
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(3, 64, 64, 256, stride=1, downsample=True)
        self.layer2 = self.make_layer(4, 256, 128, 512, stride=2, downsample=True)
        self.layer3 = self.make_layer(6, 512, 256, 1024, stride=2, downsample=True)
        self.layer4 = self.make_layer(3, 1024, 512, 2048, stride=2, downsample=True)

        # torch.nn.AdaptiveAvgPool2d(output_size = (H, W)):
        #       Applies a 2D adaptive average pooling over an input signal composed of several input planes.
        #       The output is of size H x W, for any input size. The number of output features is equal to the number of input planes.
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 5)
        )


    def make_layer(self, block_num: int, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1, downsample: bool = False):
        
        blocks = []

        blocks.append(BottleneckBlock(in_channels, mid_channels, out_channels, stride=stride, downsample=downsample))
        for i in range(block_num - 1):
            blocks.append(BottleneckBlock(out_channels, mid_channels, out_channels))
        return nn.Sequential(*blocks)

    def forward(self, x) -> Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out)

        return out