from itertools import combinations
from typing import Dict, List, Mapping, Sequence

import meerkat as mk
import numpy as np
import terra
from meerkat.contrib.gqa import read_gqa_dps
from torchvision import transforms
from tqdm import tqdm

from domino.data.gqa import ATTRIBUTE_GROUPS, DATASET_DIR, split_gqa

from . import CorrelationImpossibleError, induce_correlation

TASKS = [
    {
        "target_name": "person",
        "target_objects": [
            "person",
            "woman",
            "person",
            "baby",
            "child",
            "boy",
            "girl",
            "skier",
            "swimmer",
            "player",
        ],
        "min_h": 20,
        "min_w": 20,
        "slices": [
            {"name": "skiers", "attributes": ["skiing"], "objects": ["skier"]},
            {
                "name": "swimmers",
                "attributes": ["swimming"],
                "objects": ["swimmer"],
            },
            {
                "name": "skaters",
                "attributes": ["skating"],
                "objects": ["skaters"],
            },
            {
                "name": "surfers",
                "attributes": ["surfing"],
                "objects": ["surfer"],
            },
            {
                "name": "females",
                "attributes": ["female"],
                "objects": ["woman", "girl"],
            },
            {
                "name": "babies",
                "attributes": [],
                "objects": ["baby"],
            },
            {"name": "drinkers", "attributes": ["drinking"], "objects": []},
            {"name": "drivers", "attributes": ["driving"], "objects": ["driver"]},
            {"name": "sleepers", "attributes": ["sleeping"], "objects": []},
            {"name": "sitters", "attributes": ["sitting"], "objects": []},
        ],
    }
]


def build_rare_slice(
    target_objects: Sequence[str],
    objects: Sequence[str],
    attributes: Sequence[str],
    slice_frac: float,
    target_frac: float,
    n: int,
    min_h: int,
    min_w: int,
    split_run_id: int,
    dataset_dir: str = DATASET_DIR,
    gqa_dps: Mapping[str, mk.DataPanel] = None,
    **kwargs,
):
    dps = read_gqa_dps(dataset_dir=dataset_dir) if gqa_dps is None else gqa_dps
    attr_dp, object_dp = dps["attributes"], dps["objects"].view()

    object_dp = object_dp.lz[(object_dp["h"] > min_h) | (object_dp["w"] < min_w)]
    object_dp["target"] = object_dp["name"].isin(target_objects).astype(int)

    object_ids = mk.concat(
        (
            object_dp.lz[object_dp["name"].isin(objects)]["object_id"],
            attr_dp[attr_dp["attribute"].isin(attributes)]["object_id"],
        )
    )
    object_dp["slice"] = (
        np.isin(object_dp["object_id"], object_ids).astype(int) & object_dp["target"]
        == 1
    ).astype(int)

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

    n_pos = int(n * target_frac)
    dp = object_dp.lz[
        np.random.permutation(
            np.concatenate(
                (
                    np.random.choice(
                        np.where(object_dp["slice"] == 1)[0],
                        int(slice_frac * n_pos),
                        replace=False,
                    ),
                    np.random.choice(
                        np.where(
                            (object_dp["target"] == 1) & (object_dp["slice"] == 0),
                        )[0],
                        int((1 - slice_frac) * n_pos),
                        replace=False,
                    ),
                    np.random.choice(
                        np.where(object_dp["target"] == 0)[0], n - n_pos, replace=False
                    ),
                )
            )
        )
    ]

    return dp.merge(split_gqa.out(split_run_id, load=True), on="image_id")


@terra.Task.make_task
def collect_rare_slices(
    tasks: Sequence[dict],
    min_slice_frac: float = 0.005,
    max_slice_frac: float = 0.05,
    num_frac: int = 5,
    n: int = 40_000,
    dataset_dir: str = DATASET_DIR,
    run_dir: str = None,
):
    tasks = tasks.copy()
    dps = read_gqa_dps(dataset_dir=dataset_dir)
    attr_dp, object_dp = dps["attributes"], dps["objects"].view()

    settings = []
    for task in TASKS:
        dp = object_dp.lz[
            (object_dp["h"] > task["min_h"]) | (object_dp["w"] < task["min_w"])
        ]
        targets = dp["name"].isin(task["target_objects"]).astype(int)

        for slyce in task.pop("slices"):
            object_ids = mk.concat(
                (
                    dp.lz[dp["name"].isin(slyce["objects"])]["object_id"],
                    attr_dp[attr_dp["attribute"].isin(slyce["attributes"])][
                        "object_id"
                    ],
                )
            )
            slice = np.isin(dp["object_id"], object_ids).astype(int) & targets == 1

            target_frac = min(0.5, slice.sum() / int(max_slice_frac * n))
            assert int(target_frac * n) <= targets.sum()
            settings.extend(
                [
                    {
                        "slice_category": "rare",
                        "target_frac": target_frac,
                        "slice_frac": slice_frac,
                        "n": n,
                        **slyce,
                        **task,
                    }
                    for slice_frac in np.geomspace(
                        min_slice_frac, max_slice_frac, num_frac
                    )
                ]
            )

    return mk.DataPanel(settings)


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
    dps = read_gqa_dps(dataset_dir=dataset_dir)
    attr_dp, object_dp = dps["attributes"], dps["objects"]

    attr_dp = attr_dp.lz[attr_dp["attribute"].isin(group)]
    object_dp = object_dp.lz[np.isin(object_dp["object_id"], attr_dp["object_id"])]

    object_dp["target"] = (object_dp["name"] == target).values.astype(int)
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
    return object_dp.merge(split_gqa.out(split_run_id, load=True), on="image_id")


@terra.Task.make_task
def collect_correlation_slices(
    dataset_dir: str = DATASET_DIR,
    min_dataset_size: int = 40_000,
    max_n: int = 50_000,
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
    dps = read_gqa_dps(dataset_dir=dataset_dir)
    attr_dp, object_dp = dps["attributes"], dps["objects"]
    attr_dp = attr_dp.merge(object_dp[["object_id", "name"]], on="object_id")
    # attribute -> correlate, object -> target
    results = []
    for group_name, group in ATTRIBUTE_GROUPS.items():
        # get all objects for which at least one attribute in the group is annotated
        curr_attr_dp = attr_dp.lz[attr_dp["attribute"].isin(group)]
        curr_object_dp = object_dp.lz[
            np.isin(object_dp["object_id"], curr_attr_dp["object_id"])
        ]

        if len(curr_attr_dp) < min_dataset_size:
            continue

        df = curr_attr_dp[["attribute", "name"]].to_pandas()
        df = df[df["name"] != ""]  # filter out objects w/o name

        # only consider attribute-object pairs that have a prevalence above some
        # fraction of the entire dataset
        counts = df.value_counts()
        df = counts[counts > len(curr_attr_dp) * count_threshold_frac].reset_index()

        for attribute, name in tqdm(list(zip(df["attribute"], df["name"]))):
            dp = curr_object_dp.view()
            dp["target"] = (dp["name"] == name).values.astype(int)
            dp["correlate"] = np.isin(
                dp["object_id"],
                curr_attr_dp["object_id"][curr_attr_dp["attribute"] == attribute],
            ).astype(int)
            try:
                n = min(int(len(dp) * subsample_frac), max_n)
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
                            "target": name,
                            "group": group_name,
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


def build_noisy_label_slices(self):
    raise NotImplementedError


def buid_noisy_feature_slices(self):
    raise NotImplementedError
