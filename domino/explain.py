import meerkat as mk
import numpy as np

from domino.utils import requires_columns


# @requires_columns(dp_arg="slice_dp", columns=["emb", "slices", "target"])
def explain_slice(slice_dp: mk.DataPanel, words_dp: mk.DataPanel, slice_idx: int = 0):
    slice_mask = slice_dp["pred_slices"].data[:, slice_idx].argsort()[-10:]
    slice_proto = slice_dp.lz[slice_mask]["emb"].data.mean(axis=0)
    ref_proto = slice_dp.lz[slice_dp["target"] == 1]["emb"].data.mean(axis=0)

    words_dp["score"] = np.dot(words_dp["emb"].data.numpy(), (slice_proto - ref_proto))
    return words_dp[["word", "score", "frequency"]]
