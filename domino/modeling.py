import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet as _DenseNet
from torchvision.models import ResNet as _ResNet
from torchvision.models.densenet import _load_state_dict
from torchvision.models.densenet import model_urls as densenet_model_urls
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls as resnet_model_urls
from torchvision.models.utils import load_state_dict_from_url


class ResNet(_ResNet):

    ACTIVATION_DIMS = [64, 128, 256, 512]
    ACTIVATION_WIDTH_HEIGHT = [64, 32, 16, 8]
    RESNET_TO_ARCH = {"resnet18": [2, 2, 2, 2], "resnet50": [3, 4, 6, 3]}

    def __init__(
        self, num_classes: int, arch: str = "resnet18", pretrained: bool = True
    ):
        if arch not in self.RESNET_TO_ARCH:
            raise ValueError(
                f"config['classifier'] must be one of: {self.RESNET_TO_ARCH.keys()}"
            )

        block = BasicBlock if arch == "resnet18" else Bottleneck
        super().__init__(block, self.RESNET_TO_ARCH[arch])
        if pretrained:
            state_dict = load_state_dict_from_url(
                resnet_model_urls[arch], progress=True
            )
            self.load_state_dict(state_dict)

        self.fc = nn.Linear(512 * block.expansion, num_classes)


class DenseNet(_DenseNet):

    DENSENET_TO_ARCH = {
        "densenet121": {
            "growth_rate": 32,
            "block_config": (6, 12, 24, 16),
            "num_init_features": 64,
        }
    }

    def __init__(
        self, num_classes: int, arch: str = "densenet121", pretrained: bool = True
    ):
        if arch not in self.DENSENET_TO_ARCH:
            raise ValueError(
                f"config['classifier'] must be one of: {self.DENSENET_TO_ARCH.keys()}"
            )

        super().__init__(**self.DENSENET_TO_ARCH[arch])
        if pretrained:
            _load_state_dict(self, densenet_model_urls[arch], progress=True)

        self.classifier = nn.Linear(self.classifier.in_features, num_classes)
