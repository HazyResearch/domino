import os
import numpy as np
from functools import partial
from typing import List

import torch
from PIL import Image
import pandas as pd
from terra import Task
import torchvision.transforms as transforms
import torchvision.datasets.folder as folder
from mosaic import DataPanel, ImageColumn

from domino.utils import hash_for_split

ATTRIBUTES = [
    "5_o_clock_shadow",
    "arched_eyebrows",
    "attractive",
    "bags_under_eyes",
    "bald",
    "bangs",
    "big_lips",
    "big_nose",
    "black_hair",
    "blond_hair",
    "blurry",
    "brown_hair",
    "bushy_eyebrows",
    "chubby",
    "double_chin",
    "eyeglasses",
    "goatee",
    "gray_hair",
    "heavy_makeup",
    "high_cheekbones",
    "male",
    "mouth_slightly_open",
    "mustache",
    "narrow_eyes",
    "no_beard",
    "oval_face",
    "pale_skin",
    "pointy_nose",
    "receding_hairline",
    "rosy_cheeks",
    "sideburns",
    "smiling",
    "straight_hair",
    "wavy_hair",
    "wearing_earrings",
    "wearing_hat",
    "wearing_lipstick",
    "wearing_necklace",
    "wearing_necktie",
    "young",
]

MASK_ATTRIBUTES = [
    "background",
    "skin",
    "nose",
    "eye_g",
    "l_eye",
    "r_eye",
    "l_brow",
    "r_brow",
    "l_ear",
    "r_ear",
    "mouth",
    "u_lip",
    "l_lip",
    "hair",
    "hat",
    "ear_r",
    "neck_l",
    "neck",
    "cloth",
]


def new_celeb_transform(img: Image.Image):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda x: torch.from_numpy(np.array(x))),
            transforms.Lambda(lambda x: x.permute(2, 0, 1)),
            transforms.Lambda(lambda x: x.to(torch.float)),
        ]
    )
    return transform(img)


def celeb_mask_transform(img: Image.Image):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda x: torch.from_numpy(np.array(x))),
            transforms.Lambda(lambda x: x.max(dim=-1)[0] > 0),
        ]
    )
    return transform(img)


def celeb_transform(img: torch.Tensor):
    transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.Lambda(lambda x: x.permute(2, 0, 1)),
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda x: x.to(torch.float)),
        ]
    )
    return transform(img)


celeb_task_config = {
    "img_column": "img_path",
    "id_column": "file",
    "img_transform": celeb_transform,
    "num_classes": 2,
}

def celeb_mask_loader(filepath: str):
    if os.path.exists(filepath):
        return folder.default_loader(filepath)
    else:
        # some masks are missing
        return Image.new('RGB', (512, 512))


def get_celeb_dp(
    df: pd.DataFrame,
):
    """Build the dataframe by joining on the attribute, split and identity CelebA CSVs."""
    dp = DataPanel.from_pandas(df)
    dp.add_column("img", ImageColumn.from_filepaths(filepaths=dp["img_path"]))
    dp.add_column(
        "input",
        ImageColumn.from_filepaths(
            filepaths=dp["img_path"],
            loader=folder.default_loader,
            transform=new_celeb_transform,
        ),
    )
    for mask_attr in MASK_ATTRIBUTES:
        if f"{mask_attr}_mask_path" in dp.columns:
            dp.add_column(
                f"{mask_attr}_mask",
                ImageColumn.from_filepaths(
                    filepaths=dp[f"{mask_attr}_mask_path"],
                    transform=celeb_mask_transform,
                    loader=celeb_mask_loader,
                ),
            )
    return dp


@Task.make_task
def build_celeb_df(
    dataset_dir: str = "/afs/cs.stanford.edu/u/sabrieyuboglu/data/datasets/celeba",
    split_configs: List[dict] = None,
    split_df: str = None,
    salt: str = "abc",
    celeb_mask: bool = False,
    run_dir: str = None,
):
    """Build the dataframe by joining on the attribute, split and identity CelebA CSVs."""
    identity_df = pd.read_csv(
        os.path.join(dataset_dir, "identity_CelebA.txt"),
        delim_whitespace=True,
        header=None,
        names=["file", "identity"],
    )
    attr_df = pd.read_csv(
        os.path.join(dataset_dir, "list_attr_celeba.txt"),
        delim_whitespace=True,
        header=1,
    )
    attr_df.columns = pd.Series(attr_df.columns).apply(lambda x: x.lower())
    attr_df = ((attr_df + 1) // 2).rename_axis("file").reset_index()

    celeb_df = identity_df.merge(attr_df, on="file", validate="one_to_one")

    celeb_df["img_path"] = celeb_df.file.apply(
        lambda x: os.path.join(dataset_dir, "img_align_celeba", x)
    )

    if celeb_mask:
        mask_dir = os.path.join(dataset_dir, "CelebAMask-HQ")
        mask_df = pd.read_csv(
            os.path.join(mask_dir, "CelebA-HQ-to-CelebA-mapping.txt"), delimiter=r"\s+"
        )[["idx", "orig_file"]]
        for mask_attr in MASK_ATTRIBUTES:
            mask_df[f"{mask_attr}_mask_path"] = mask_df.idx.apply(
                lambda idx: os.path.join(
                    mask_dir,
                    "CelebAMask-HQ-mask-anno",
                    str(idx // 2000),
                    f"{str(idx).zfill(5)}_{mask_attr}.png",
                )
            )
        celeb_df = celeb_df.merge(mask_df, left_on="file", right_on="orig_file")

        celeb_df["img_path"] = celeb_df.idx.apply(
            lambda x: os.path.join(
                dataset_dir, "CelebAMask-HQ/CelebA-HQ-img", f"{x}.jpg"
            )
        )
    else:
        celeb_df["img_path"] = celeb_df.file.apply(
            lambda x: os.path.join(dataset_dir, "img_align_celeba", x)
        )

    if split_df is not None:
        return celeb_df.merge(split_df[["file", "split"]], on="file", how="left")
    else:
        # add splits by hashing each file name to a number between 0 and 1
        if split_configs is None:
            split_configs = [{"split": "train", "size": len(celeb_df)}]

        # hash on identity to avoid same person straddling the train-test divide
        example_hash = celeb_df.identity.apply(partial(hash_for_split, salt=salt))
        total_size = sum([config["size"] for config in split_configs])

        if total_size > len(celeb_df):
            raise ValueError("Total size cannot exceed full dataset size.")

        start = 0
        celeb_df["example_hash"] = example_hash
        dfs = []
        for config in split_configs:
            frac = config["size"] / total_size
            end = start + frac
            df = celeb_df[(start < example_hash) & (example_hash <= end)]
            induce_kwargs = config.get("induce_kwargs", None)
            if induce_kwargs is not None:
                df = induce_correlation(df, n=config["size"], **induce_kwargs)
            elif len(df) > config["size"]:
                df = df.sample(n=config["size"])
            df["split"] = config["split"]
            dfs.append(df)
            start = end

        return pd.concat(dfs)


def induce_correlation(
    df: pd.DataFrame,
    corr: float,
    n: int,
    attr_a: str,
    attr_b: str,
    mu_a: float = None,
    mu_b: float = None,
    match_mu: bool = False,
    replace: bool = False,
):
    """
    Induce a correlation `corr` between two boolean columns `attr_a` and `attr_b` by
    subsampling `df`, while maintaining mean and variance. If `match_mu` is `True` then
    take the minimum mean among the two attributes and use it for both
    """
    if mu_a is None:
        mu_a = df[attr_a].mean()

    if mu_b is None:
        mu_b = df[attr_b].mean()

    if match_mu:
        mu = min(mu_a, mu_b)
        mu_a, mu_b = mu, mu

    var_a = (mu_a) * (1 - mu_a)  # df[attr_a].var()
    var_b = (mu_b) * (1 - mu_b)  # df[attr_b].var()
    n_a1 = mu_a * n
    n_b1 = mu_b * n

    n_1 = (n_a1 * n_b1 / n) + corr * np.sqrt(var_a * var_b * n ** 2)

    if (n_1 > n_a1) or (n_1 > n_b1) or n_1 < 0:
        raise ValueError(
            f"Cannot achieve correlation of {corr} while maintaining means for "
            f"attributes {attr_a=} and {attr_b=}."
        )

    both1 = (df[attr_a] == 1) & (df[attr_b] == 1)

    indices = []
    indices.extend(np.random.choice(np.where(both1)[0], size=int(n_1), replace=replace))
    indices.extend(
        np.random.choice(
            np.where(df[attr_a] & (1 - both1))[0], size=int(n_a1 - n_1), replace=replace
        )
    )
    indices.extend(
        np.random.choice(
            np.where(df[attr_b] & (1 - both1))[0], size=int(n_b1 - n_1), replace=replace
        )
    )

    indices.extend(
        np.random.choice(
            np.where((df[attr_a] == 0) & (df[attr_b] == 0))[0],
            size=n - len(indices),
            replace=replace,
        )
    )

    return df.iloc[indices]
