import os

import PIL
import gradio as gr
import numpy as np
from mosaic import ListColumn, DataPanel, ImageColumn, ImagePath, NumpyArrayColumn

import pandas as pd
import numpy as np
from torch._C import Value
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import normalize


def solicit_feedback_imgs(
    dp: DataPanel,
    img_column: str = "img",
    label_column: str = "label",
    size: tuple = (224, 224),
    overwrite: bool = False,
    rank_by: str = None,
    num_examples: int = 100,
    run_dir: str = None,
):

    # TODO: find a better solution for this
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = "_feedback_cache"
    os.makedirs(run_dir, exist_ok=True)

    # prepare examples
    if rank_by is not None:
        example_idxs = (-dp[rank_by]).argsort()[:num_examples]
    else:
        example_idxs = np.random.randint(0, len(dp), num_examples)

    is_img_col = isinstance(dp[img_column], ImageColumn)

    examples = []
    for rank, example_idx in enumerate(example_idxs):
        example_idx = int(example_idx)
        label = dp[label_column][example_idx]
        if False and is_img_col:
            image: ImagePath = dp[img_column][example_idx]
            image_path = image.filepath

        else:
            image = dp[img_column][example_idx]
            image.resize(size)
            if not isinstance(image, PIL.Image.Image):
                raise ValueError(
                    "`img_column` must either be an `ImageColumn` or materialize to `PIL.Image`"
                )
            image_path = os.path.join(run_dir, f"image_{example_idx}.jpg")
            image.save(image_path)

        examples.append(
            [
                rank,
                example_idx,
                str(label),  # important this is a str, gradio hangs otherwise
                image_path,
                dp["feedback_label"][example_idx] if "feedback_label" in dp else None,
            ]
        )

    label_dp = dp[["image_id"]].lz[example_idxs]
    label_dp["feedback_label"] = NumpyArrayColumn(["unlabeled"] * len(example_idxs))
    label_dp["feedback_pos_mask"] = ListColumn([None] * len(example_idxs))
    label_dp["feedback_neg_mask"] = ListColumn([None] * len(example_idxs))

    # define feedback function
    def submit_feedback(rank, example_idx, label, img, feedback_label):
        pos_mask = (img == np.array([0, 169, 255])).all(axis=-1)
        neg_mask = (img == np.array([255, 64, 64])).all(axis=-1)
        label_dp["feedback_label"][rank] = feedback_label
        label_dp["feedback_pos_mask"][rank] = pos_mask
        label_dp["feedback_neg_mask"][rank] = neg_mask
        return []

    iface = gr.Interface(
        submit_feedback,
        [
            gr.inputs.Number(),
            gr.inputs.Number(),
            gr.inputs.Textbox(),
            gr.inputs.Image(shape=size),
            gr.inputs.Radio(choices=["positive", "negative", "abstain"]),
        ],
        outputs=[],
        examples=examples,
        layout="vertical",
    )
    return iface.launch(inbrowser=False, inline=False), label_dp


def merge_in_feedback(
    base_dp: DataPanel, feedback_dp: DataPanel, on: str = "file", remove: bool = False
):
    size = feedback_dp["feedback_pos_mask"].shape[1:]
    if remove:
        if "feedback_label" in base_dp.column_names:
            base_dp.remove_column("feedback_label")
        if "feedback_pos_mask" in base_dp.column_names:
            base_dp.remove_column("feedback_pos_mask")
        if "feedback_neg_mask" in base_dp.column_names:
            base_dp.remove_column("feedback_neg_mask")

    if "feedback_label" not in base_dp.column_names:
        base_dp.add_column(
            "feedback_label", NumpyArrayColumn(["unlabeled"] * len(base_dp))
        )
    if "feedback_pos_mask" not in base_dp.column_names:
        base_dp.add_column(
            "feedback_pos_mask",
            NumpyArrayColumn(np.zeros((len(base_dp), *size))),
        )
    if "feedback_neg_mask" not in base_dp.column_names:
        base_dp.add_column(
            "feedback_neg_mask",
            NumpyArrayColumn(np.zeros((len(base_dp), *size))),
        )

    for batch in feedback_dp.batch(batch_size=1, num_workers=0):
        index = np.where(base_dp[on] == batch[on])
        base_dp["feedback_label"][index[0]] = batch["feedback_label"]
        base_dp["feedback_neg_mask"][index[0]] = batch["feedback_neg_mask"]
        base_dp["feedback_pos_mask"][index[0]] = batch["feedback_pos_mask"]
    return base_dp


class ScribbleModel:
    def __init__(
        self, threshold: float = 0.2, strategy="pos", activation_size: int = (7, 7)
    ):
        self.lr = LogisticRegression()
        self.threshold = threshold
        self.strategy = strategy
        self.activation_size = activation_size

    def prepare(self):
        pass

    def fit(self, train_dp: DataPanel, activation_col: str = "activation"):
        kernel_size = (
            train_dp["feedback_pos_mask"].shape[-2] // self.activation_size[0],
            train_dp["feedback_pos_mask"].shape[-1] // self.activation_size[1],
        )
        if self.strategy == "mask_pos_v_neg":
            pooled_masks = {
                f"{feedback_mask}_pool": nn.functional.avg_pool2d(
                    input=train_dp[feedback_mask].to_tensor().to(float),
                    kernel_size=kernel_size,
                ).numpy()
                for feedback_mask in ["feedback_neg_mask", "feedback_pos_mask"]
            }
            y = (
                pooled_masks["feedback_pos_mask_pool"].flatten()
                - pooled_masks["feedback_neg_mask_pool"].flatten()
            )
            mask = ((y > self.threshold) + (y < -self.threshold)).astype(bool)
            y = (y > 0)[mask]

            activation_dim = train_dp[activation_col].shape[1]
            x = (
                train_dp[activation_col]
                .transpose(0, 2, 3, 1)
                .reshape(-1, activation_dim)
            )
            x = x[mask]

        elif self.strategy == "mask_pos_v":
            pos_mask_pool = nn.functional.avg_pool2d(
                input=train_dp["feedback_pos_mask"].to_tensor().to(float),
                kernel_size=kernel_size,
            ).numpy()
            activation_dim = train_dp[activation_col].shape[1]
            x = (
                train_dp[activation_col]
                .transpose(0, 2, 3, 1)
                .reshape(-1, activation_dim)
            )
            y = pos_mask_pool.flatten() > self.threshold

        elif self.strategy == "example":
            y = (train_dp["feedback_label"].data == "positive").astype(int)
            x = train_dp[activation_col].data.mean(axis=(2, 3))

        x = normalize(x)
        self.lr.fit(x, y)

    def predict(
        self,
        test_dp: DataPanel,
        activation_col: str = "activation",
        reduction: str = "max",
    ):
        if self.strategy in ["mask_pos_v", "mask_pos_v_neg"]:
            activation_dim = test_dp[activation_col].shape[1]
            x = (
                test_dp[activation_col]
                .transpose(0, 2, 3, 1)
                .reshape(-1, activation_dim)
            )
        else:
            x = test_dp[activation_col].data.mean(axis=(2, 3))

        x = normalize(x)
        y = self.lr.predict_proba(x)

        if self.strategy in ["mask_pos_v", "mask_pos_v_neg"]:

            y = y[:, 1].reshape(-1, *self.activation_size)
            if reduction == "max":
                return y.max(axis=(1, 2))
            elif reduction == "mean":
                return y.mean(axis=(1, 2))
            elif reduction is None:
                return y
            else:
                raise ValueError()
        else:
            return y[:, 1]
