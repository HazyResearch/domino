import meerkat as mk
import numpy as np
from scipy.stats import mode


def describe_slice(
    data: mk.DataPanel = None,
    embeddings: str = "embedding",
    slices: str = "slices",
    text: mk.DataPanel = None,
    text_embeddings: str = "embedding",
    phrase: str = "output_phrase",
    slice_idx: int = 0,
    slice_threshold: float = 0.5
):
    slice_mask = data[slices].data[:, slice_idx] > slice_threshold
    slice_data = data.lz[slice_mask]
    slice_proto = slice_data[embeddings].data.mean(axis=0)
    mode_target = mode(slice_data["target"].data).mode
    ref_proto = data.lz[data["target"] == 1]["emb"].data.mean(axis=0)

    text["score"] = np.dot(text[text_embeddings], (slice_proto - ref_proto))
    return text[[phrase, "score"]]
