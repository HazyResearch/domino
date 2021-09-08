import io
import os
from collections import OrderedDict
from functools import partial
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import meerkat as mk
import numpy as np
import PIL
import requests
import terra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from mpl_toolkits.axes_grid1 import ImageGrid
from torch import nn
from torch._C import Value
from torch.autograd import Variable
from torchvision import models, transforms

from domino.utils import nested_getattr
from domino.vision import score

# code here is taken primarily from this colab https://colab.research.google.com/github/google-research/big_transfer/blob/master/colabs/big_transfer_pytorch.ipynb


def transform(img: PIL.Image.Image):

    transform = tv.transforms.Compose(
        [
            tv.transforms.Resize(
                (128, 128), interpolation=tv.transforms.InterpolationMode.BILINEAR
            ),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    return transform(img)


@terra.Task
def embed_images(
    dp: mk.DataPanel,
    img_column: str,
    layers: Mapping[str, nn.Module] = None,
    reduction_name: Sequence[str] = "mean",
    batch_size: int = 128,
    num_workers: int = 4,
    model: str = "resnet50",
    mmap: bool = False,
    device: int = 0,
    run_dir: str = None,
    **kwargs,
) -> mk.DataPanel:
    weights = get_weights(
        "BiT-M-R50x1"
    )  # You could use other variants, such as R101x3 or R152x4 here, but it is not advisable in a colab.

    model = ResNetV2(ResNetV2.BLOCK_UNITS["r50"], width_factor=1)
    model.load_from(weights)

    col = dp[img_column].to_lambda(transform)

    if layers is None:
        layers = {"body": "body"}
    layers = {name: nested_getattr(model, layer) for name, layer in layers.items()}

    def _get_reduction_fn(reduction_name):
        if reduction_name == "max":
            reduction_fn = partial(torch.mean, dim=[-1, -2])
        elif reduction_name == "mean":
            reduction_fn = partial(torch.mean, dim=[-1, -2])
        else:
            raise ValueError(f"reduction_fn {reduction_name} not supported.")
        reduction_fn.__name__ = reduction_name
        return reduction_fn

    class ActivationExtractor:
        """Class for extracting activations a targetted intermediate layer"""

        def __init__(self, reduction_fn: callable = None):
            self.activation = None
            self.reduction_fn = reduction_fn

        def add_hook(self, module, input, output):
            if self.reduction_fn is not None:
                output = self.reduction_fn(output)
            self.activation = output

    layer_to_extractor = {}

    for name, layer in layers.items():
        extractor = ActivationExtractor(reduction_fn=_get_reduction_fn(reduction_name))
        layer.register_forward_hook(extractor.add_hook)
        layer_to_extractor[name] = extractor

    @torch.no_grad()
    def _score(batch: mk.TensorColumn):
        x = batch.data.to(device)
        model(x)  # Run forward pass

        return {
            name: extractor.activation.cpu().detach().numpy()
            for name, extractor in layer_to_extractor.items()
        }

    model.to(device)
    emb_dp = col.map(
        _score,
        pbar=True,
        is_batched_fn=True,
        batch_size=batch_size,
        num_workers=num_workers,
        mmap=mmap,
        mmap_path=os.path.join(run_dir, "emb_mmap.npy"),
        flush_size=128,
    )
    return mk.concat([dp, emb_dp], axis=1)


def get_weights(bit_variant):
    response = requests.get(
        f"https://storage.googleapis.com/bit_models/{bit_variant}.npz"
    )
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(
        cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups
    )


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


def tf2th(conv_weights):
    """Possibly convert HWIO to OIHW"""
    if conv_weights.ndim == 4:
        conv_weights = np.transpose(conv_weights, [3, 2, 0, 1])
    return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
    """
    Follows the implementation of "Identity Mappings in Deep Residual Networks" here:
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original ResNetv2 has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride)

    def forward(self, x):
        # Conv'ed branch
        out = self.relu(self.gn1(x))

        # Residual branch
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(out)

        # The first block has already applied pre-act before splitting, see Appendix.
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual

    def load_from(self, weights, prefix=""):
        with torch.no_grad():
            self.conv1.weight.copy_(
                tf2th(weights[prefix + "a/standardized_conv2d/kernel"])
            )
            self.conv2.weight.copy_(
                tf2th(weights[prefix + "b/standardized_conv2d/kernel"])
            )
            self.conv3.weight.copy_(
                tf2th(weights[prefix + "c/standardized_conv2d/kernel"])
            )
            self.gn1.weight.copy_(tf2th(weights[prefix + "a/group_norm/gamma"]))
            self.gn2.weight.copy_(tf2th(weights[prefix + "b/group_norm/gamma"]))
            self.gn3.weight.copy_(tf2th(weights[prefix + "c/group_norm/gamma"]))
            self.gn1.bias.copy_(tf2th(weights[prefix + "a/group_norm/beta"]))
            self.gn2.bias.copy_(tf2th(weights[prefix + "b/group_norm/beta"]))
            self.gn3.bias.copy_(tf2th(weights[prefix + "c/group_norm/beta"]))
            if hasattr(self, "downsample"):
                self.downsample.weight.copy_(
                    tf2th(weights[prefix + "a/proj/standardized_conv2d/kernel"])
                )
        return self


class ResNetV2(nn.Module):
    BLOCK_UNITS = {
        "r50": [3, 4, 6, 3],
        "r101": [3, 4, 23, 3],
        "r152": [3, 8, 36, 3],
    }

    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            3, 64 * wf, kernel_size=7, stride=2, padding=3, bias=False
                        ),
                    ),
                    ("padp", nn.ConstantPad2d(1, 0)),
                    ("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
                    # The following is subtly not the same!
                    # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=64 * wf, cout=256 * wf, cmid=64 * wf
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=256 * wf, cout=256 * wf, cmid=64 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=256 * wf,
                                            cout=512 * wf,
                                            cmid=128 * wf,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=512 * wf, cout=512 * wf, cmid=128 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=512 * wf,
                                            cout=1024 * wf,
                                            cmid=256 * wf,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=1024 * wf, cout=1024 * wf, cmid=256 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block4",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=1024 * wf,
                                            cout=2048 * wf,
                                            cmid=512 * wf,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=2048 * wf, cout=2048 * wf, cmid=512 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[3] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )

        self.zero_head = zero_head
        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("gn", nn.GroupNorm(32, 2048 * wf)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("avg", nn.AdaptiveAvgPool2d(output_size=1)),
                    ("conv", nn.Conv2d(2048 * wf, head_size, kernel_size=1, bias=True)),
                ]
            )
        )

    def forward(self, x):
        x = self.head(self.body(self.root(x)))
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0]

    def load_from(self, weights, prefix="resnet/"):
        with torch.no_grad():
            self.root.conv.weight.copy_(
                tf2th(weights[f"{prefix}root_block/standardized_conv2d/kernel"])
            )
            self.head.gn.weight.copy_(tf2th(weights[f"{prefix}group_norm/gamma"]))
            self.head.gn.bias.copy_(tf2th(weights[f"{prefix}group_norm/beta"]))
            if self.zero_head:
                nn.init.zeros_(self.head.conv.weight)
                nn.init.zeros_(self.head.conv.bias)
            else:
                self.head.conv.weight.copy_(
                    tf2th(weights[f"{prefix}head/conv2d/kernel"])
                )
                self.head.conv.bias.copy_(tf2th(weights[f"{prefix}head/conv2d/bias"]))

            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f"{prefix}{bname}/{uname}/")
        return self
