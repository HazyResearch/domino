import os
from functools import partial
from typing import Mapping, Sequence

import meerkat as mk
import terra
import torch
from torch import nn
from torch._C import Value
from torchvision import models, transforms

from domino.vision import score

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@terra.Task
def embed_images(
    dp: mk.DataPanel,
    img_column: str,
    layers: Mapping[str, nn.Module],
    reduction_fn: Sequence[callable] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    model: str = "resnet50",
    mmap: bool = False,
    device: int = 0,
    run_dir: str = None,
    **kwargs,
) -> mk.DataPanel:
    if model == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Model {model} not supported.")

    col = dp[img_column].to_lambda(transform)

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
        extractor = ActivationExtractor(reduction_fn=torch.mean)
        layer.register_forward_hook(extractor.add_hook)
        layer_to_extractor[name] = extractor

    @torch.no_grad()
    def _score(batch: mk.TensorColumn):
        x = batch.data.to(device)
        model(x)  # Run forward pass

        return {
            **{
                name: extractor.activation.cpu().detach().numpy()
                for name, extractor in layer_to_extractor.items()
            },
        }

    dp["emb"] = col.map(
        _score,
        pbar=True,
        is_batched_fn=True,
        batch_size=batch_size,
        num_workers=num_workers,
        mmap=mmap,
        mmap_path=os.path.join(run_dir, "emb_mmap.npy"),
        flush_size=128,
    )
    return dp
