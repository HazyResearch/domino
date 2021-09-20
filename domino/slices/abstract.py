from dataclasses import dataclass
from typing import Mapping

import meerkat as mk
import numpy as np
import terra

from domino.utils import merge_in_split

from .utils import synthesize_preds


def _get_slice_builder(dataset: str):
    if dataset == "imagenet":
        from .imagenet import ImageNetSliceBuilder

        sb = ImageNetSliceBuilder()
    elif dataset == "eeg":
        from .eeg import EegSliceBuilder

        sb = EegSliceBuilder()
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return sb


@terra.Task
def build_setting(dataset: str, slice_category: str, data_dp: mk.DataPanel, **kwargs):
    sb = _get_slice_builder(dataset=dataset)
    return sb.build_setting(data_dp=data_dp, slice_category=slice_category, **kwargs)


@terra.Task
def collect_settings(
    dataset: str, slice_category: str, data_dp: mk.DataPanel, **kwargs
):
    sb = _get_slice_builder(dataset=dataset)
    settings_dp = sb.collect_settings(
        data_dp=data_dp, slice_category=slice_category, **kwargs
    )
    settings_dp["setting_id"] = np.arange(len(settings_dp))
    return settings_dp


class AbstractSliceBuilder:
    def __init__(self, config: dict = None, **kwargs):
        pass

    def build_setting(
        self,
        data_dp: mk.DataPanel,
        slice_category: str,
        split_dp: mk.DataPanel,
        synthetic_preds: bool = False,
        synthetic_kwargs: Mapping[str, object] = None,
        **kwargs,
    ) -> mk.DataPanel:
        if slice_category == "correlation":
            dp = self.build_correlation_setting(data_dp=data_dp, **kwargs)
        elif slice_category == "rare":
            dp = self.build_rare_setting(data_dp=data_dp, **kwargs)
        else:
            raise ValueError(f"Slice category '{slice_category}' not recognized.")

        if synthetic_preds:
            synthetic_kwargs = {} if synthetic_kwargs is None else synthetic_kwargs
            dp["pred"] = synthesize_preds(dp, **synthetic_kwargs)

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
