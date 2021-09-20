import datetime
from dataclasses import dataclass
from typing import Union

import meerkat as mk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import binary_cross_entropy
from tqdm import tqdm

from domino.utils import VariableColumn, requires_columns

from .abstract import SliceDiscoveryMethod


class SpotlightSDM(SliceDiscoveryMethod):
    @dataclass
    class Config(SliceDiscoveryMethod.Config):
        emb: str = "emb"
        min_weight: int = 100
        num_steps: int = 1000
        learning_rate: float = 1e-2
        device: Union[str, int] = 0

    RESOURCES_REQUIRED = {"cpu": 1, "gpu": 1}

    def __init__(self, config: dict = None, **kwargs):
        super().__init__(config, **kwargs)
        self.means = []
        self.precisions = []

    @requires_columns(
        dp_arg="data_dp", columns=[VariableColumn("self.config.emb"), "pred", "target"]
    )
    def fit(
        self,
        data_dp: mk.DataPanel,
        model: nn.Module = None,
    ):
        all_weights = []
        weights_unnorm = None
        losses = binary_cross_entropy(
            torch.tensor(data_dp["pred"].data).to(torch.float32),
            torch.tensor(data_dp["target"]).to(torch.float32),
            reduction="none",
        )
        for slice_idx in range(self.config.n_slices):
            if slice_idx != 0:
                weights_unnorm /= max(weights_unnorm)
                losses *= 1 - weights_unnorm

            (weights, weights_unnorm, mean, log_precision) = run_spotlight(
                embeddings=torch.tensor(data_dp[self.config.emb].data),
                losses=losses,
                min_weight=self.config.min_weight,
                barrier_x_schedule=np.geomspace(
                    len(data_dp) - self.config.min_weight,
                    0.05 * self.config.min_weight,
                    self.config.num_steps,
                ),
                learning_rate=self.config.learning_rate,
                device=self.config.device,
            )
            self.means.append(mean.cpu().detach())
            self.precisions.append(log_precision.cpu().detach())
            all_weights.append(weights)
        return self

    @requires_columns(dp_arg="data_dp", columns=[VariableColumn("self.config.emb")])
    def transform(
        self,
        data_dp: mk.DataPanel,
    ):
        losses = binary_cross_entropy(
            torch.tensor(data_dp["pred"]).to(torch.float),
            torch.tensor(data_dp["target"]).to(torch.float),
            reduction="none",
        ).to(torch.float)
        dp = data_dp.view()
        all_weights = []

        for slice_idx in range(self.config.n_slices):
            weights, _, _, _ = md_adversary_weights(
                mean=self.means[slice_idx],
                precision=torch.eye(self.means[slice_idx].shape[0])
                * torch.exp(self.precisions[slice_idx]),
                x=torch.tensor(data_dp[self.config.emb].data),
                losses=losses,
            )
            all_weights.append(weights.numpy())
        dp["pred_slices"] = np.stack(all_weights, axis=1)
        return dp


## Source below from spotlight implementation https://github.com/gregdeon/spotlight/blob/main/torch_spotlight/spotlight.py


def gaussian_probs(mean, precision, x):
    # Similarity kernel: describe how similar each point in x is to mean as number in [0, 1]
    # - mean: (dims) vector
    # - precision: (dims, dims) precision matrix; must be PSD
    # - x: (num_points, dims) set of points
    dists = torch.sum(((x - mean) @ precision) * (x - mean), axis=1)
    return torch.exp(-dists / 2)


def md_adversary_weights(mean, precision, x, losses, counts=None):
    # Calculate normalized weights, average loss, and spotlight size for current mean and precision settings
    # - mean, precision, x: as in gaussian_probs
    # - losses: (num_points) vector of losses
    # - counts: (num_points) vector of number of copies of each point to include. defaults to all-ones.

    if counts is None:
        counts = torch.ones_like(losses)

    weights_unnorm = gaussian_probs(mean, precision, x)
    total_weight = weights_unnorm @ counts
    weights = weights_unnorm / total_weight
    weighted_loss = (weights * counts) @ losses

    return (weights, weights_unnorm, weighted_loss, total_weight)


def md_objective(
    mean,
    precision,
    x,
    losses,
    min_weight,
    barrier_x,
    barrier_scale,
    flip_objective=False,
    counts=None,
    labels=None,
    label_coeff=0.0,
    predictions=None,
    prediction_coeff=0.0,
):
    # main objective
    weights, _, weighted_loss, total_weight = md_adversary_weights(
        mean, precision, x, losses
    )
    if flip_objective:
        weighted_loss = -weighted_loss

    # barrier
    if total_weight < (min_weight + barrier_x):
        barrier_penalty = (
            barrier_scale
            * (total_weight - (min_weight + barrier_x)) ** 2
            / barrier_x ** 2
        )
        weighted_loss -= barrier_penalty

    # regularization
    if labels is not None:
        categories = torch.arange(max(labels) + 1).reshape(-1, 1)
        label_probs = (labels == categories).float() @ weights
        label_entropy = torch.distributions.Categorical(
            probs=label_probs
        ).entropy() / np.log(2)
        weighted_loss -= label_coeff * label_entropy
    if predictions is not None:
        categories = torch.arange(max(predictions) + 1).reshape(-1, 1)
        prediction_probs = (predictions == categories).float() @ weights
        prediction_entropy = torch.distributions.Categorical(
            probs=prediction_probs
        ).entropy() / np.log(2)
        weighted_loss -= prediction_coeff * prediction_entropy

    return (weighted_loss, total_weight)


class ResetOnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def _reduce_lr(self, epoch):
        super(ResetOnPlateau, self)._reduce_lr(epoch)
        self._reset()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def run_spotlight(
    embeddings,
    losses,
    min_weight,
    barrier_x_schedule,
    barrier_scale=1,
    learning_rate=1e-3,
    scheduler_patience=20,
    scheduler_decay=0.5,
    print_every=200,
    device=0,
    flip_objective=False,
    labels=None,
    counts=None,
    label_coeff=0.0,
    predictions=None,
    prediction_coeff=0.0,
):
    x = embeddings.clone().to(torch.float).to(device=device)
    y = losses.clone().to(device=device)
    dimensions = x.shape[1]

    mean = torch.zeros((dimensions,), requires_grad=True, device=device)

    log_precision = torch.tensor(np.log(0.0001), requires_grad=True, device=device)
    optimizer = optim.Adam([mean, log_precision], lr=learning_rate)

    scheduler = ResetOnPlateau(
        optimizer, patience=scheduler_patience, factor=scheduler_decay
    )

    num_steps = len(barrier_x_schedule)

    objective_history = []
    total_weight_history = []
    lr_history = []

    start_time = datetime.datetime.now()

    for t in tqdm(range(num_steps)):
        optimizer.zero_grad()
        precision = torch.exp(log_precision)
        precision_matrix = torch.eye(x.shape[1], device=device) * precision

        objective, total_weight = md_objective(
            mean,
            precision_matrix,
            x,
            y,
            min_weight,
            barrier_x_schedule[t],
            barrier_scale,
            flip_objective,
            counts,
            labels,
            label_coeff,
            predictions,
            prediction_coeff,
        )
        neg_objective = -objective
        neg_objective.backward()
        optimizer.step()
        scheduler.step(neg_objective)

        objective_history.append(objective.detach().cpu().item())
        total_weight_history.append(total_weight.detach().cpu().item())
        lr_history.append(get_lr(optimizer))

        if (t + 1) % print_every == 0:

            precision_matrix = torch.eye(
                dimensions, device=precision.device
            ) * torch.exp(log_precision)

            weights, weights_unnorm, weighted_loss, total_weight = md_adversary_weights(
                mean, precision_matrix, x, y
            )

    final_weights = weights.detach().cpu().numpy()
    final_weights_unnorm = weights_unnorm.detach().cpu().numpy()

    return (
        final_weights,
        final_weights_unnorm,
        mean,
        log_precision,
    )
