import torch
import torch.nn as nn
import torch.hub

__all__ = ['r2plus1d_34_32_ig65m']

model_urls = {
    'r2plus1d_34_32_ig65m': 'https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth'
}
class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, mid_planes, stride=1, padding=1):
        super(Conv2Plus1D, self).__init__(nn.Conv3d(in_planes, mid_planes, kernel_size=(1,3,3),
                                                    stride=(1,stride,stride), padding=(0,padding,padding),
                                                    bias=False),
                                          nn.BatchNorm3d(mid_planes),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(mid_planes, out_planes, kernel_size=(3,1,1),
                                                    stride=(stride,1,1), padding=(padding,0,0),
                                                    bias=False))
    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, conv_builder, stride=1, downsample=None):
        mid_planes = (in_planes * planes * 3 * 3 * 3) // (in_planes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(in_planes, planes, mid_planes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, mid_planes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        mid_planes = (in_planes * planes * 3 * 3 * 3) // (in_planes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, mid_planes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class R2Plus1dStem(nn.Sequential):
    def __init__(self):
        super(R2Plus1dStem, self).__init__(nn.Conv3d(3, 45, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False),
                                           nn.BatchNorm3d(45),
                                           nn.ReLU(inplace=True),
                                           nn.Conv3d(45, 64, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False),
                                           nn.BatchNorm3d(64),
                                           nn.ReLU(inplace=True))
class VideoResNet(nn.Module):
    def __init__(self, block, conv_makers, layers, stem, num_classes=400, zero_init_residual=False):
        super(VideoResNet, self).__init__()
        self.in_planes = 64
        self.stem = stem()
        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # init weights
        self._initialize_weights()
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)
        return x
    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.in_planes, planes, conv_builder, stride, downsample))

        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, conv_builder))

        return nn.Sequential(*layers)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def r2plus1d_34_32_ig65m(num_classes, pretrained=False, progress=False):
    """R(2+1)D 34-layer IG65M model for clips of length 32 frames.
    Args:
      num_classes: Number of classes in last classification layer
      pretrained: If True, loads weights pretrained on 65 million Instagram videos
      progress: If True, displays a progress bar of the download to stderr
    """
    assert not pretrained or num_classes == 359, 'pretrained on 359 classes'
    return r2plus1d_34(num_classes=num_classes, arch='r2plus1d_34_32_ig65m', pretrained=pretrained, progress=progress)

def r2plus1d_34(num_classes, pretrained=False, progress=False, arch=None):
    model = VideoResNet(block=BasicBlock, conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 6, 3], stem=R2Plus1dStem)
    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model