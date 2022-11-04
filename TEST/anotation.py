import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
import collections
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
from torchsummary import summary
from torch import optim


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1, padding=None):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      groups=groups, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1, padding=None,
                 act_layer=nn.ReLU):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      groups=groups, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            act_layer(inplace=True) if act_layer is not None else None,
        )


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.reduction_block = ConvBNReLU(inplanes, width, kernel_size=1)
        self.conv_block = ConvBNReLU(width, width, kernel_size=3, stride=stride,
                                     groups=groups, dilation=dilation)

        self.expansion_block = ConvBN(width, planes * self.expansion, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.reduction_block(x)
        out = self.conv_block(out)
        out = self.expansion_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity
        out = self.rele(out)

        return out


class MaxBottleneck(nn.Module):
    expansion: int = 4

    def __init__(self,
                 inplances: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 ) -> None:
        super(MaxBottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups

        self.reduction_block = ConvBNReLU(inplances, width, kernel_size=1)
        self.conv_block = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.expansion_block = ConvBN(width, planes * self.expansion, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.reduction_block(x)
        out = self.conv_block(out)

        out = self.expansion_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EffBottleneck(nn.Module):
    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 ) -> None:
        super(EffBottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups

        if stride == 2:
            self.reduction_block = ConvBNReLU(inplanes, width, kernel_size=1)
            self.conv_block = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),
                                            ConvBNReLU(width, width, kernel_size=3, stride=1,
                                                       groups=groups, dilation=dilation))
        else:
            self.reduction_block = ConvBNReLU(inplanes, width, kernel_size=1)
            self.conv_block = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        self.expansion_block = ConvBN(width, planes * self.expansion, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.reduction_block(x)
        out = self.conv_block(out)

        out = self.expansion_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block: Type[Union[Bottleneck, MaxBottleneck, EffBottleneck]],
                 layers: List[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 width_mult: float = 1.0, ) -> None:
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.block_idx = 0
        self.net_block_idx = sum(layers)

        self.conv1 = ConvBNReLU(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, int(64 * width_mult), layers[0], stride=1, dilate=False)
        self.layer2 = self._make_layer(block, int(128 * width_mult), layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(256 * width_mult), layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(512 * width_mult), layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * width_mult) * block.expansion, num_classes)

    def _make_layer(self, block: Type[Union[Bottleneck, MaxBottleneck, EffBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride)

        layers = []
        for block_idx in range(blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            layers.append(block(self.inplanes, planes, groups=self.groups, stride=stride, downsample=downsample,
                                base_width=self.base_width, dilation=self.dilation))
            self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet50(pretrain=False, layer_setting=[3, 4, 6, 3], width_mult: float = 1.0, **kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, layers=layer_setting, width_mult=width_mult, **kwargs)


def resnet50_max(pretrained=False, layer_setting=[3, 4, 6, 3], width_mult: float = 1.0, **kwargs: Any) -> ResNet:
    return ResNet(MaxBottleneck, layers=layer_setting, width_mult=width_mult, **kwargs)


def resnet50_hybrid(pretrained=False, layer_setting=[3, 4, 6, 3], width_mult: float = 1.0, **kwargs: Any) -> ResNet:
    return ResNet(EffBottleneck, layers=layer_setting, width_mult=width_mult, **kwargs)


def get_mean(dataset):
    meanRGB = [np.mean(image.numpy(), axis=(1, 2)) for image, _ in dataset]
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])
    return [meanR, meanG, meanB]


def get_std(dataset):
    stdRGB = [np.std(image.numpy(), axis=(1,2)) for image, _ in dataset]
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    return [stdR, stdG, stdB]


if __name__ == '__main__':
    path2data = 'C:\\Users\\become\\Desktop\\개인파일\\ResDataset'

    train_ds = datasets.STL10(path2data, split='train', download=False, transform=transforms.ToTensor())
    test_ds = datasets.STL10(path2data, split='test', download=False, transform=transforms.ToTensor())

    train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(get_mean(train_ds), get_std(train_ds))])
    test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(get_mean(test_ds), get_std(test_ds))])
    train_ds.transform = train_transforms
    test_ds.transform = test_transforms

    train_dataloader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False)


    '''
    y_train = [y for _, y in train_ds]
    counter_train = collections.Counter(y_train)
    print(counter_train)
    
    np.random.seed(0)

    def show(img, y=None, color=True):
        npimg = img.numpy()
        npimt_tr = np.transpose(npimg, (1, 2, 0))

        plt.imshow(npimt_tr)
        if y is not None:
            plt.title('labels: ' + str(y))
        plt.show()
    grid_size = 4
    rnd_inds = np.random.randint(0, len(train_ds), grid_size)
    print('image indices: ', rnd_inds)

    x_grid = [train_ds[i][0] for i in rnd_inds]
    y_grid = [train_ds[i][1] for i in rnd_inds]

    x_grid = utils.make_grid(x_grid, nrow=4, padding=1)
    print(x_grid.shape)

    plt.figure(figsize=(10.0, 10.0))
    show(x_grid, y_grid)
    '''

    model = resnet50_max().eval()
    summary(model, (3, 128, 128))

    input = torch.randn(1, 3, 224, 224)
    output = model(input)

    loss = output.sum()
    loss.backward()
    print('Checked a single forward/backward iteration')
