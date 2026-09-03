"""ResNet-18 feature backbone of PSD-Few.

The encoder is split into per-stage submodules (``layer1``..``layer4``) so that
individual stages can be addressed during fine-tuning.
"""

import torch.nn as nn
import torchvision.models as models


class ResNet18Backbone(nn.Module):
    """ImageNet pre-trained ResNet-18 feature encoder."""

    def __init__(self, pretrained: bool = True):
        """Create the ResNet-18 stages with optional ImageNet weights."""
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        """Map a batch of images to the final convolutional feature map."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
