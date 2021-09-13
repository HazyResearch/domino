from dataclasses import dataclass
from typing import Mapping

import meerkat as mk
import terra
from torch._C import Value

from . import synthesize_preds


@terra.Task.make(no_dump_args={"gqa_dps"})
def build_slice():
    pass


class AbstractSliceBuilder:
    def __init__(self, config: dict = None, **kwargs):
        pass

    def build_slice(
        self,
        slice_category: str,
        split_dp: mk.DataPanel,
        synthetic_preds: bool = False,
        synthetic_kwargs: Mapping[str, object] = None,
        **kwargs,
    ) -> mk.DataPanel:
        if slice_category == "correlation":
            dp = self.build_correlation_slice(**kwargs)
        elif slice_category == "rare":
            dp = self.build_rare_slice(**kwargs)
        else:
            raise ValueError(f"Slice category '{slice_category}' not recognized.")

        if synthetic_preds:
            synthetic_kwargs = {} if synthetic_kwargs is None else synthetic_kwargs
            dp["pred"] = synthesize_preds(dp, **synthetic_kwargs)

        return dp.merge(split_dp, on="image_id")

    def build_correlation_slices(self):
        raise NotImplementedError

    def build_rare_slices(self):
        raise NotImplementedError

    def build_noisy_label_slices(self):
        raise NotImplementedError

    def buid_noisy_feature_slices(self):
        raise NotImplementedError
