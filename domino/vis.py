import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from domino.bss import SourceSeparator
import torch
from scipy.ndimage import gaussian_filter
from terra import Task
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from mosaic import DataPanel
from domino.data.celeb import celeb_transform
from torchvision import transforms


@Task.make_task
def visualize_components(
    data_df: pd.DataFrame,
    separator: SourceSeparator = None,
    components: torch.Tensor = None,
    sort_comp_idx: int = 0,
    split: str = "valid",
    batch_size: int = 256,
    num_workers: int = 10,
    num_examples: int = 10,
    device: int = 0,
    run_dir: str = None,
):
    """[summary]

    Args:
        data_df (pd.DataFrame): [description]
        separator (SourceSeparator, optional): [description]. Defaults to None.
        components (torch.Tensor, optional): With shape (num_examples, height * width,
            num_components). Defaults to None.
        sort_comp_idx (int, optional): [description]. Defaults to 0.
        split (str, optional): [description]. Defaults to "valid".
        batch_size (int, optional): [description]. Defaults to 256.
        num_workers (int, optional): [description]. Defaults to 10.
        num_examples (int, optional): [description]. Defaults to 10.
        device (int, optional): [description]. Defaults to 0.
        run_dir (str, optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # prepare dataset and dataloader
    dataset = Dataset.load_image_dataset(
        data_df[data_df.split == split].to_dict("records"), img_columns="img_path"
    )

    if (separator is None) == (components is None):
        raise ValueError("Must pass either `separator` or `components`, but not both.")

    if separator is not None:
        dl = dataset.to_dataloader(
            columns=["img_path"],
            column_to_transform={"img_path": celeb_transform},
            batch_size=batch_size,
            num_workers=num_workers,
        )
        components, _, _, _ = separator.compute_components(dl, device=device)

    components = components.numpy()

    #  sort examples by component get top and bottom num_examples // 2
    example_idxs = []
    comp_max = components.max(axis=1)
    example_idxs.extend(comp_max[:, sort_comp_idx].argsort()[: num_examples // 2])
    example_idxs.extend((-comp_max[:, sort_comp_idx]).argsort()[: num_examples // 2])

    # unpack height and width
    n_components = components.shape[-1]
    width = int(np.sqrt(components.shape[1]))
    components = components.reshape(-1, width, width, n_components)
    fig, ax = plt.subplots(
        n_components + 1,
        num_examples,
        figsize=(3 * num_examples, 15),
        subplot_kw={"xticks": [], "yticks": []},
    )
    for col, example_idx in enumerate(example_idxs):
        # get input image
        inp = dataset[int(example_idx)]["img_path"]
        inp = celeb_transform(inp).to(torch.int)
        inp = inp.detach().cpu().numpy().squeeze()
        # plot colored image
        ax[0, col].imshow(inp.transpose(1, 2, 0))
        # this vector transforms the image to grayscale
        inp = np.dot(inp.transpose(1, 2, 0), np.array([0.2125, 0.7154, 0.0721]))

        for comp_idx in range(n_components):
            curr_ax = ax[comp_idx + 1, col]
            # visualize
            curr_components = components[example_idx, :, :, comp_idx].squeeze()
            curr_max_comp = curr_components.max()
            curr_components = curr_components.repeat(
                inp.shape[0] / curr_components.shape[0], axis=0
            ).repeat(inp.shape[1] / curr_components.shape[1], axis=1)

            curr_ax.imshow(inp, cmap="gray")

            comp_abs = np.absolute(curr_components)
            alpha = comp_abs / comp_abs.max()
            vmin = components[:, :, :, comp_idx].min()
            vmax = components[:, :, :, comp_idx].max()
            curr_ax.imshow(
                curr_components, alpha=alpha, cmap="PiYG", vmin=vmin, vmax=vmax
            )
            curr_ax.set_title(f"max_value={curr_max_comp:.2f}")

    # need color bar for orig image row to make spacing work
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="PiYG"),
        ax=ax[0, :],
        fraction=0.05,
    )
    for comp_idx in range(n_components):
        vmin = components[:, :, :, comp_idx].min()
        vmax = components[:, :, :, comp_idx].max()
        curr_ax = ax[comp_idx + 1, :]
        fig.colorbar(
            plt.cm.ScalarMappable(cmap="PiYG"),
            ax=curr_ax,
            boundaries=np.linspace(vmin, vmax, 10000),
            fraction=0.05,
        )

    # label rows with component
    for row_ax, row in zip(
        ax[:, 0], ["orig image"] + [f"comp_{idx}" for idx in range(n_components)]
    ):
        row_ax.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-row_ax.yaxis.labelpad - 5, 0),
            xycoords=row_ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )
    # label col with example filename
    for col_ax, col in zip(ax[0, :], data_df.iloc[example_idxs].file):
        col_ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, 5),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )

    return data_df.iloc[example_idxs]


def visualize_component_dp(
    dp: DataPanel,
    comp_idx: int = 0,
    class_idx: int = 1,
    num_examples: int = 10,
    flip: bool = False,
    target_column: str = "y",
    probs_column: str = "probs",
    components_column: str = "components",
    image_column: str = "raw_input",
    id_column: str = "img_filename",
    run_dir: str = None,
    **kwargs,
):
    preds = torch.tensor(dp[probs_column])[:, class_idx]
    targets = (torch.tensor(dp[target_column]) == class_idx).to(int)
    components = torch.tensor(dp[components_column])
    components = components.view(components.shape[0], -1, components.shape[-1])

    #components = components - components.mean(axis=1, keepdim=True)  # todo try removing
    components = components.numpy()

    # flip the order of the component
    components = (-1 * components) if flip else components

    #  prepare subplots
    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(constrained_layout=True, figsize=(3 * num_examples, 20))
    gs = GridSpec(5, num_examples, figure=fig, height_ratios=[1.5, 1, 1, 1, 1])
    ax = [
        [
            fig.add_subplot(gs[row + 1, col], xticks=[], yticks=[])
            for col in range(num_examples)
        ]
        for row in range(4)
    ]

    # add colorbar
    global_comp_min = components[:, :, comp_idx].min()
    global_comp_max = components[:, :, comp_idx].max()
    colorbar_ax = fig.add_subplot(gs[0, -1])
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="PiYG"),
        cax=colorbar_ax,
        boundaries=np.linspace(global_comp_min, global_comp_max, 10000),
        pad=0,
    )

    # get maximum value of the component in each example
    # plot distribution of max component values across examples
    example_wise_comp_max = components.mean(axis=1)[:, comp_idx]
    hist_ax = fig.add_subplot(gs[0, 0:3])
    sns.histplot(data=example_wise_comp_max, ax=hist_ax)
    hist_ax.set_xlabel("maximum component value")
    hist_ax.set_ylabel("# of examples")

    # plot preds scatter
    if preds is not None:
        scatter_ax = fig.add_subplot(gs[0, 3:6])
        sns.scatterplot(x=example_wise_comp_max, y=preds, ax=scatter_ax)
        scatter_ax.set_xlabel("maximum component value")
        scatter_ax.set_ylabel("model prediction")

    roc_ax = fig.add_subplot(gs[0, 6:9])
    if preds is not None and targets is not None:
        percentiles = np.linspace(0, 100, num=6)
        thresholds = np.percentile(example_wise_comp_max, q=percentiles)
        for lower, upper, p in zip(thresholds[:-1], thresholds[1:], percentiles[1:]):
            mask = (example_wise_comp_max >= lower) * (example_wise_comp_max < upper)
            curr_targets = targets[mask]
            if curr_targets.min() == curr_targets.max():
                continue
            fpr, tpr, _ = roc_curve(curr_targets, preds[mask])
            score = roc_auc_score(curr_targets, preds[mask])
            roc_ax.plot(fpr, tpr, label=f"{int(p)}th percentile, auroc: {score:0.2f}")
            plt.legend()

    #  sort examples by component get top and bottom examples
    examples_sorted_by_comp = (-1 * example_wise_comp_max).argsort()

    # unpack height and width
    n_components = components.shape[-1]
    width = int(np.sqrt(components.shape[1]))
    components = components.reshape(-1, width, width, n_components)

    for row, (name, example_idxs) in enumerate(
        [
            ("largest", examples_sorted_by_comp[:num_examples]),
            ("smallest", examples_sorted_by_comp[-num_examples:]),
        ]
    ):
        for col, example_idx in enumerate(example_idxs):
            # get input image
            example = dp[int(example_idx)]
            img = example[image_column]
            # TODO: figure out way for this to work for all datasets
            # img = img_transform(img).to(torch.int)
            img = transforms.Resize([128, 128])(img)
            img = transforms.ToTensor()(img)
            img = img.detach().cpu().numpy().squeeze()
            img_ax = ax[row * 2][col]
            is_color = img.shape[0] == 1
            if is_color:
                img = img.transpose(1, 2, 0)
                # transform the image to grayscale
                gray_img = np.dot(img, np.array([0.2125, 0.7154, 0.0721]))
                img_ax.imshow(img)
            else:
                img = gray_img = img.squeeze()
                img_ax.imshow(img, cmap="gray")
            
            # label with filename
            img_ax.annotate(
                example[id_column][:10],
                xy=(0.5, 1),
                xytext=(0, 5),
                xycoords="axes fraction",
                textcoords="offset points",
                size="large",
                ha="center",
                va="baseline",
            )

            # visualize component
            curr_ax = ax[row * 2 + 1][col]
            curr_components = components[example_idx, :, :, comp_idx].squeeze()
            curr_max_comp = curr_components.max()
            curr_components = curr_components.repeat(
                img.shape[0] / curr_components.shape[0], axis=0
            ).repeat(img.shape[1] / curr_components.shape[1], axis=1)

            curr_ax.imshow(gray_img, cmap="gray")

            comp_abs = np.absolute(curr_components)
            alpha = comp_abs / comp_abs.max()
            curr_ax.imshow(
                curr_components,
                alpha=alpha,
                cmap="PiYG",
                vmin=global_comp_min,
                vmax=global_comp_max,
            )
            curr_ax.set_title(f"max_value={curr_max_comp:.2f}", fontsize=15)

        # label row
        row_ax = ax[row * 2][0]
        row_ax.annotate(
            name,
            xy=(0, 0.5),
            xytext=(-row_ax.yaxis.labelpad - 5, 0),
            xycoords=row_ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )

    if run_dir is not None:
        plt.savefig(os.path.join(run_dir, "out.pdf"))

    # return data_df.iloc[example_idxs]


@Task.make_task
def visualize_component(
    data_df: pd.DataFrame,
    img_column: str,
    target_column: str,
    id_column: str,
    components: torch.Tensor,
    preds: torch.Tensor = None,
    targets: torch.tensor = None,
    comp_idx: int = 0,
    split: str = "valid",
    num_examples: int = 10,
    flip: bool = False,
    img_transform: callable = celeb_transform,
    run_dir: str = None,
    **kwargs,
):
    # prepare dataset and dataloader
    dataset = Dataset.load_image_dataset(
        data_df[data_df.split == split].to_dict("records"), img_columns="img_path"
    )

    if len(preds.shape) > 1:
        preds = torch.softmax(preds, dim=-1)[:, -1]

    components = components - components.mean(axis=1, keepdim=True)  # todo try removing
    components = components.numpy()

    # flip the order of the component
    components = (-1 * components) if flip else components

    #  prepare subplots
    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(constrained_layout=True, figsize=(3 * num_examples, 20))
    gs = GridSpec(5, num_examples, figure=fig, height_ratios=[1.5, 1, 1, 1, 1])
    ax = [
        [
            fig.add_subplot(gs[row + 1, col], xticks=[], yticks=[])
            for col in range(num_examples)
        ]
        for row in range(4)
    ]

    # add colorbar
    global_comp_min = components[:, :, comp_idx].min()
    global_comp_max = components[:, :, comp_idx].max()
    colorbar_ax = fig.add_subplot(gs[0, -1])
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="PiYG"),
        cax=colorbar_ax,
        boundaries=np.linspace(global_comp_min, global_comp_max, 10000),
        pad=0,
    )

    # get maximum value of the component in each example
    # plot distribution of max component values across examples
    example_wise_comp_max = components.max(axis=1)[:, comp_idx]
    hist_ax = fig.add_subplot(gs[0, 0:3])
    sns.histplot(data=example_wise_comp_max, ax=hist_ax)
    hist_ax.set_xlabel("maximum component value")
    hist_ax.set_ylabel("# of examples")

    # plot preds scatter
    if preds is not None:
        scatter_ax = fig.add_subplot(gs[0, 3:6])
        sns.scatterplot(x=example_wise_comp_max, y=preds, ax=scatter_ax)
        scatter_ax.set_xlabel("maximum component value")
        scatter_ax.set_ylabel("model prediction")

    roc_ax = fig.add_subplot(gs[0, 6:9])
    if preds is not None and targets is not None:
        percentiles = np.linspace(0, 100, num=6)
        thresholds = np.percentile(example_wise_comp_max, q=percentiles)
        for lower, upper, p in zip(thresholds[:-1], thresholds[1:], percentiles[1:]):
            mask = (example_wise_comp_max >= lower) * (example_wise_comp_max < upper)
            curr_targets = targets[mask]
            if curr_targets.min() == curr_targets.max():
                continue
            fpr, tpr, _ = roc_curve(curr_targets, preds[mask])
            score = roc_auc_score(curr_targets, preds[mask])
            roc_ax.plot(fpr, tpr, label=f"{int(p)}th percentile, auroc: {score:0.2f}")
            plt.legend()

    #  sort examples by component get top and bottom examples
    examples_sorted_by_comp = (-1 * example_wise_comp_max).argsort()

    # unpack height and width
    n_components = components.shape[-1]
    width = int(np.sqrt(components.shape[1]))
    components = components.reshape(-1, width, width, n_components)

    for row, (name, example_idxs) in enumerate(
        [
            ("largest", examples_sorted_by_comp[:num_examples]),
            ("smallest", examples_sorted_by_comp[-num_examples:]),
        ]
    ):
        for col, example_idx in enumerate(example_idxs):
            # get input image
            example = dataset[int(example_idx)]
            inp = example[img_column]
            # TODO: figure out way for this to work for all datasets
            # inp = img_transform(inp).to(torch.int)
            inp = transforms.ToTensor()(inp)
            inp = inp.detach().cpu().numpy().squeeze()

            # plot colored image
            img_ax = ax[row * 2][col]
            img_ax.imshow(inp.transpose(1, 2, 0))
            # label with filename
            img_ax.annotate(
                example[id_column][:10],
                xy=(0.5, 1),
                xytext=(0, 5),
                xycoords="axes fraction",
                textcoords="offset points",
                size="large",
                ha="center",
                va="baseline",
            )

            # transform the image to grayscale
            inp = np.dot(inp.transpose(1, 2, 0), np.array([0.2125, 0.7154, 0.0721]))

            # visualize component
            curr_ax = ax[row * 2 + 1][col]
            curr_components = components[example_idx, :, :, comp_idx].squeeze()
            curr_max_comp = curr_components.max()
            curr_components = curr_components.repeat(
                inp.shape[0] / curr_components.shape[0], axis=0
            ).repeat(inp.shape[1] / curr_components.shape[1], axis=1)

            curr_ax.imshow(inp, cmap="gray")

            comp_abs = np.absolute(curr_components)
            alpha = comp_abs / comp_abs.max()
            curr_ax.imshow(
                curr_components,
                alpha=alpha,
                cmap="PiYG",
                vmin=global_comp_min,
                vmax=global_comp_max,
            )
            curr_ax.set_title(f"max_value={curr_max_comp:.2f}", fontsize=15)

        # label row
        row_ax = ax[row * 2][0]
        row_ax.annotate(
            name,
            xy=(0, 0.5),
            xytext=(-row_ax.yaxis.labelpad - 5, 0),
            xycoords=row_ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )

    if run_dir is not None:
        plt.savefig(os.path.join(run_dir, "out.pdf"))

    return data_df.iloc[example_idxs]


@Task.make_task
def compute_comp_corr(
    df: pd.DataFrame,
    components: torch.Tensor,
    split: str = None,
    columns_to_drop: List[str] = None,
    columns_to_keep: List[str] = None,
    reduction: str = "max",
    plot: bool = True,
    run_dir=None,
):
    """Compute the correlation between attributes in a dataset (e.g. CelebA)."""

    if split is not None:
        df = df[df.split == split]
    if columns_to_drop is None:
        columns_to_drop = ["split", "path", "file", "identity", "example_hash"]
    df = df.drop(columns=columns_to_drop)

    if columns_to_keep is not None:
        df = df[columns_to_keep]

    # add components to dataframe
    if reduction == "max":
        components = components.max(dim=1)[0]
    elif reduction == "min":
        components = components.min(dim=1)[0]
    else:
        raise ValueError(f"Reduction {reduction} not supported.")
    for comp_idx in range(components.shape[-1]):
        df[f"comp_{comp_idx}"] = components[:, comp_idx]

    corr = df.corr()

    if plot:
        # Generate a custom diverging colormap
        plt.figure(figsize=(20, 20))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatma
        cg = sns.clustermap(
            corr,
            cmap=cmap,
            square=True,
            vmax=1.0,
            linewidths=0.5,
            xticklabels=1,
            yticklabels=1,
        )
        cg.ax_row_dendrogram.set_visible(False)
        cg.ax_col_dendrogram.set_visible(False)

        if run_dir is not None:
            plt.savefig(os.path.join(run_dir, "cluster_plot.pdf"))
    return corr.reset_index()


@Task.make_task
def component_scatterplot(
    df: pd.DataFrame,
    components: torch.Tensor,
    split: str = None,
    reduction: str = "max",
    plot: bool = True,
    run_dir=None,
):
    """Compute the correlation between attributes in a dataset (e.g. CelebA)."""

    if split is not None:
        df = df[df.split == split]

    # add components to dataframe
    if reduction == "max":
        components = components.max(dim=1)[0]
    elif reduction == "min":
        components = components.min(dim=1)[0]
    else:
        raise ValueError(f"Reduction {reduction} not supported.")

    # add components to dataframe
    for comp_idx in range(components.shape[-1]):
        df[f"comp_{comp_idx}"] = components[:, comp_idx]

    if plot:
        # Generate a custom diverging colormap
        plt.figure(figsize=(20, 20))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatma
        cg = sns.clustermap(
            corr,
            cmap=cmap,
            square=True,
            vmax=1.0,
            linewidths=0.5,
            xticklabels=1,
            yticklabels=1,
        )
        cg.ax_row_dendrogram.set_visible(False)
        cg.ax_col_dendrogram.set_visible(False)

        if run_dir is not None:
            plt.savefig(os.path.join(run_dir, "cluster_plot.pdf"))
    return df


def get_saliency(
    model: torch.nn.Module,
    inp: torch.tensor,
    device: int = 0,
    show=False,
    threshold: float = 0.5,
    sigma: int = 5,
    ax=None,
) -> torch.Tensor:
    """Compute a saliency map for `model` on `inp`.

    Args:
        model (torch.nn.Module): a trained image model
        inp (torch.tensor): input to the model of shape (channels, height, width).
        device (int, optional): cuda device. Defaults to 1.
        show (bool, optional): show the slaiency map the . Defaults to False.
        threshold (float, optional): standardized saliencies below this value are
            clipped to zero. Defaults to 0.5.
        sigma (int, optional): the standard deviation for the gaussian kernel. Defaults
            to 5.
        ax ([type], optional): axis for plotting. Defaults to None.

    Returns:
        [torch.Tensor]: a saliency tensor of shape (height, width)
    """
    model.eval()
    model.to(device)

    # move to GPU if available
    inp.requires_grad = True
    inp = inp.to(device)

    # add batch dimension
    inp = inp.unsqueeze(0)
    out = model(inp)

    (grad,) = torch.autograd.grad(out[:, -1], inp)

    def standardize_gradient(x):
        x = torch.abs(x.detach().cpu().squeeze())
        x = torch.sum(x, dim=0)
        x -= x.min()
        x /= x.max()
        return x

    grad = standardize_gradient(grad)

    # put on range 0 - 1

    grad[grad < threshold] = 0
    if show:
        inp = inp.detach().cpu().numpy().squeeze()
        inp = np.dot(inp.transpose(1, 2, 0), np.array([0.2125, 0.7154, 0.0721]))

        grad_to_show = grad.numpy()
        grad_to_show = gaussian_filter(grad_to_show, [sigma] * 2, mode="constant")
        grad_to_show /= grad_to_show.max()
        if ax is None:
            plt.imshow(inp, cmap="gray")
            plt.imshow(grad_to_show, alpha=grad_to_show)
        else:
            ax.imshow(inp, cmap="gray")
            ax.imshow(grad_to_show, alpha=grad_to_show)
        return grad_to_show

    return grad
