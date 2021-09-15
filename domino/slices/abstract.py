from dataclasses import dataclass
from typing import Mapping

import meerkat as mk
import terra
from torch._C import Value

from domino.utils import merge_in_split

from . import synthesize_preds


@terra.Task
def build_setting(dataset: str, data_dp: mk.DataPanel, **kwargs):
    if dataset == "imagenet":
        from .imagenet import ImageNetSliceBuilder

        sb = ImageNetSliceBuilder()
    elif dataset == "eeg":
        from .eeg import EegSliceBuilder

        sb = EegSliceBuilder()
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    return sb.build_setting(data_dp=data_dp, **kwargs)


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
            dp = self.build_correlation_slices(data_dp=data_dp, **kwargs)
        elif slice_category == "rare":
            dp = self.build_rare_slices(data_dp=data_dp, **kwargs)
        else:
            raise ValueError(f"Slice category '{slice_category}' not recognized.")

        if synthetic_preds:
            synthetic_kwargs = {} if synthetic_kwargs is None else synthetic_kwargs
            dp["pred"] = synthesize_preds(dp, **synthetic_kwargs)

        dp = merge_in_split(dp, split_dp)
        return dp

    def build_correlation_slices(self) -> mk.DataPanel:
        raise NotImplementedError

    def build_rare_slices(self) -> mk.DataPanel:
        raise NotImplementedError

    def build_noisy_label_slices(self) -> mk.DataPanel:
        raise NotImplementedError

    def buid_noisy_feature_slices(self) -> mk.DataPanel:
        raise NotImplementedError
