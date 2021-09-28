from dataclasses import dataclass
from typing import Dict, List, Mapping
from uuid import uuid4

import meerkat as mk
import numpy as np
import terra
import torch

from domino.utils import merge_in_split

from .utils import synthesize_preds


def _get_slice_builder(dataset: str):
    if dataset == "imagenet":
        from .imagenet import ImageNetSliceBuilder

        sb = ImageNetSliceBuilder()
    elif dataset == "eeg":
        from .eeg import EegSliceBuilder

        sb = EegSliceBuilder()
    elif dataset == "celeba":
        from .celeba import CelebASliceBuilder

        sb = CelebASliceBuilder()
    elif dataset == "mimic":
        from .mimic import MimicSliceBuilder

        sb = MimicSliceBuilder()
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return sb


@terra.Task
def build_setting(
    dataset: str,
    slice_category: str,
    data_dp: mk.DataPanel,
    split_dp: mk.DataPanel,
    build_setting_kwargs: Dict,
    synthetic_preds: bool = False,
    synthetic_kwargs: Mapping[str, object] = None,
):
    sb = _get_slice_builder(dataset=dataset)
    dp = sb.build_setting(
        slice_category=slice_category,
        data_dp=data_dp,
        split_dp=split_dp,
        **build_setting_kwargs,
    )
    if synthetic_preds:
        synthetic_kwargs = {} if synthetic_kwargs is None else synthetic_kwargs
        preds = synthesize_preds(dp, **synthetic_kwargs)
        dp["probs"] = torch.tensor([1 - preds, preds]).T
    return dp


@terra.Task
def collect_settings(
    dataset: str, slice_category: str, data_dp: mk.DataPanel, **kwargs
):
    sb = _get_slice_builder(dataset=dataset)
    settings_dp = sb.collect_settings(
        data_dp=data_dp, slice_category=slice_category, **kwargs
    )
    settings_dp["setting_id"] = [str(uuid4()) for _ in range(len(settings_dp))]
    return settings_dp


@terra.Task
def concat_settings(setting_dps: List[mk.DataPanel]):
    return mk.concat(setting_dps, axis="rows")


class AbstractSliceBuilder:
    def __init__(self, config: dict = None, **kwargs):
        pass

    def build_setting(
        self,
        data_dp: mk.DataPanel,
        slice_category: str,
        split_dp: mk.DataPanel,
        **kwargs,
    ) -> mk.DataPanel:
        if slice_category == "correlation":
            dp = self.build_correlation_setting(data_dp=data_dp, **kwargs)
        elif slice_category == "rare":
            dp = self.build_rare_setting(data_dp=data_dp, **kwargs)
        elif slice_category == "noisy_label":
            dp = self.build_noisy_label_setting(data_dp=data_dp, **kwargs)
        elif slice_category == "noisy_feature":
            dp = self.build_noisy_feature_setting(data_dp=data_dp, **kwargs)
        else:
            raise ValueError(f"Slice category '{slice_category}' not recognized.")

        dp = merge_in_split(dp, split_dp)
        return dp

    def build_correlation_setting(self) -> mk.DataPanel:
        raise NotImplementedError

    def build_rare_setting(self) -> mk.DataPanel:
        raise NotImplementedError

    def build_noisy_label_setting(self) -> mk.DataPanel:
        raise NotImplementedError

    def build_noisy_feature_setting(self) -> mk.DataPanel:
        raise NotImplementedError

    def collect_settings(
        self,
        data_dp: mk.DataPanel,
        slice_category: str,
        **kwargs,
    ) -> mk.DataPanel:
        if slice_category == "correlation":
            dp = self.collect_correlation_settings(data_dp=data_dp, **kwargs)
        elif slice_category == "rare":
            dp = self.collect_rare_settings(data_dp=data_dp, **kwargs)
        elif slice_category == "noisy_label":
            dp = self.collect_noisy_label_settings(data_dp=data_dp, **kwargs)
        elif slice_category == "noisy_feature":
            dp = self.collect_noisy_feature_settings(data_dp=data_dp, **kwargs)
        else:
            raise ValueError(f"Slice category '{slice_category}' not recognized.")

        return dp

    def collect_correlation_settings(self) -> mk.DataPanel:
        raise NotImplementedError

    def collect_rare_settings(self) -> mk.DataPanel:
        raise NotImplementedError

    def collect_noisy_label_settings(self) -> mk.DataPanel:
        raise NotImplementedError

    def collect_noisy_feature_settings(self) -> mk.DataPanel:
        raise NotImplementedError
