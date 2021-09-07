import uuid

import meerkat as mk
import numpy as np
import terra
from torchvision import transforms
from tqdm import tqdm

from domino.data.visual_genome import ATTRIBUTE_GROUPS, DATASET_DIR, read_vg, split_vg

from . import CorrelationImpossibleError, induce_correlation


def build_correlation_slice(
    target: str,
    correlate: str,
    group: str,
    corr: float,
    n: int,
    dataset_dir: str = DATASET_DIR,
    split_run_id: int = None,
    **kwargs,
):
    group = ATTRIBUTE_GROUPS[group]
    if correlate not in group:
        raise ValueError("")
    _, attr_dp, object_dp = read_vg(dataset_dir=dataset_dir)

    attr_dp = attr_dp.lz[attr_dp["attribute"].isin(group)]
    object_dp = object_dp.lz[np.isin(object_dp["object_id"], attr_dp["object_id"])]

    object_dp["target"] = (object_dp["syn_name"] == target).values.astype(int)
    object_dp["correlate"] = np.isin(
        object_dp["object_id"],
        attr_dp["object_id"][attr_dp["attribute"] == correlate],
    ).astype(int)

    indices = induce_correlation(
        dp=object_dp,
        corr=corr,
        attr_a="target",
        attr_b="correlate",
        match_mu=True,
        n=n,
    )

    preprocessing = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    object_dp["input"] = object_dp["object_image"].to_lambda(preprocessing)
    object_dp["id"] = object_dp["object_id"]

    object_dp = object_dp.lz[indices]

    # merge in split
    return object_dp.merge(split_vg.out(split_run_id, load=True), on="image_id")


@terra.Task
def collect_correlation_slices(
    dataset_dir: str = DATASET_DIR,
    min_dataset_size: int = 40_000,
    min_corr: float = 0,
    max_corr: float = 0.8,
    num_corr: int = 5,
    subsample_frac: float = 0.3,
    count_threshold_frac: float = 0.002,
    run_dir: str = None,
):
    """
    One challenge with using the attributes in visual genome is that annotators are free
    to label whatever attributes they choose. So, if an object isn't labeled with an
    attribute, it doesn't necessarily mean that it doesn't have that attribute – the
    annotator may have just chosen not to mention it. In other words, it's clear when
    the attribute is  present, but unclear when it's not. We address this is by forming
    groups of mutually exclusive attributes: {"long", "short"}, {"blue", "green", "red}.
    The assumption we then make is that if any one of attributes is labeled for an
    object, then the rest of the attributes in the group are False.
    """
    _, attr_dp, object_dp = read_vg(dataset_dir=dataset_dir)

    # attribute -> correlate, object -> target
    results = []
    for name, group in ATTRIBUTE_GROUPS.items():
        # get all objects for which at least one attribute in the group is annotated
        curr_attr_dp = attr_dp.lz[attr_dp["attribute"].isin(group)]
        curr_object_dp = object_dp.lz[
            np.isin(object_dp["object_id"], curr_attr_dp["object_id"])
        ]

        if len(curr_attr_dp) < min_dataset_size:
            continue

        df = curr_attr_dp[["attribute", "syn_name"]].to_pandas()
        df = df[df["syn_name"] != ""]  # filter out objects w/o syn_name

        # only consider attribute-object pairs that have a prevalence above some
        # fraction of the entire dataset
        counts = df.value_counts()
        df = counts[counts > len(curr_attr_dp) * count_threshold_frac].reset_index()

        for attribute, syn_name in tqdm(list(zip(df["attribute"], df["syn_name"]))):
            dp = curr_object_dp.view()
            dp["target"] = (dp["syn_name"] == syn_name).values.astype(int)
            dp["correlate"] = np.isin(
                dp["object_id"],
                curr_attr_dp["object_id"][curr_attr_dp["attribute"] == attribute],
            ).astype(int)
            try:
                n = int(len(dp) * subsample_frac)
                for corr in [
                    min_corr,
                    max_corr,
                ]:
                    _ = induce_correlation(
                        dp=dp,
                        corr=corr,
                        attr_a="target",
                        attr_b="correlate",
                        match_mu=True,
                        n=n,
                    )

                results.extend(
                    [
                        {
                            "correlate": attribute,
                            "target": syn_name,
                            "group": name,
                            "corr": corr,
                            "n": n,
                        }
                        for corr in np.linspace(min_corr, max_corr, num_corr)
                    ]
                )
            except CorrelationImpossibleError:
                pass
    return mk.DataPanel(results)

    # object -> correlate, object -> target


def build_rare_slices(self):
    raise NotImplementedError


def build_noisy_label_slices(self):
    raise NotImplementedError


def buid_noisy_feature_slices(self):
    raise NotImplementedError
