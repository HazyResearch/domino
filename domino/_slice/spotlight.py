from typing import Union

import meerkat as mk
import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import cross_entropy

from domino.utils import unpack_args

from .abstract import Slicer


class SpotlightSlicer(Slicer):

    r"""
    Slice a dataset with The Spotlight algorithm [deon_2022]_.

    TODO: add docstring similar to the Domino one

    .. [deon_2022]

        d’Eon, G., d’Eon, J., Wright, J. R. & Leyton-Brown, K.
        The Spotlight: A General Method for Discovering Systematic Errors in Deep
        Learning Models. arXiv:2107. 00758 [cs, stat] (2021)
    """

    def __init__(
        self,
        spotlight_size: int = 0.02,  # recommended from paper
        num_steps: int = 1000,
        learning_rate: float = 1e-3,  # default from the implementation
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config.spotlight_size = spotlight_size
        self.config.num_steps = num_steps
        self.config.learning_rate = learning_rate

        self.means = []
        self.precisions = []

    def _compute_losses(
        self, targets: np.ndarray = "target", pred_probs: np.ndarray = "pred_probs"
    ):
        return cross_entropy(
            pred_probs,
            targets,
            reduction="none",
        )

    def fit(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ):
        embeddings, targets, pred_probs = unpack_args(
            data, embeddings, targets, pred_probs
        )

        pred_probs = torch.tensor(pred_probs).to(torch.float32)
        targets = torch.tensor(targets).to(torch.long)

        all_weights = []
        weights_unnorm = None
        losses = self._compute_losses(pred_probs=pred_probs, targets=targets)

        min_weight = targets.shape[0] * self.config.spotlight_size
        for slice_idx in range(self.config.n_slices):
            if slice_idx != 0:
                weights_unnorm /= max(weights_unnorm)
                losses *= 1 - weights_unnorm

            (weights, weights_unnorm, mean, log_precision) = run_spotlight(
                embeddings=embeddings,
                losses=losses,
                min_weight=min_weight,
                barrier_x_schedule=np.geomspace(  #
                    targets.shape[0] - min_weight,
                    0.05 * min_weight,
                    self.config.num_steps,
                ),
                learning_rate=self.config.learning_rate,
                device=self.config.device,
            )
            self.means.append(mean.cpu().detach())
            self.precisions.append(log_precision.cpu().detach())
            all_weights.append(weights)
        return self

    def predict_proba(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> np.ndarray:
        embeddings, targets, pred_probs = unpack_args(
            data, embeddings, targets, pred_probs
        )

        pred_probs = torch.tensor(pred_probs).to(torch.float32)
        targets = torch.tensor(targets).to(torch.long)

        losses = self._compute_losses(pred_probs=pred_probs, targets=targets)

        all_weights = []

        for slice_idx in range(self.config.n_slices):
            weights, _, _, _ = md_adversary_weights(
                mean=self.means[slice_idx],
                precision=torch.eye(self.means[slice_idx].shape[0])
                * torch.exp(self.precisions[slice_idx]),
                x=embeddings,
                losses=losses,
            )
            all_weights.append(weights.numpy())
        return np.stack(all_weights, axis=1)

    def predict(
        self,
        data: mk.DataPanel,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> np.ndarray:
        probs = self.predict_proba(
            data=data,
            embeddings=embeddings,
            targets=targets,
            pred_probs=pred_probs,
        )

        # TODO (Greg): check if this is the preferred way to get hard predictions from
        # probabilities
        return (probs > 0.5).astype(np.int32)


# Source below copied from spotlight implementation
# https://github.com/gregdeon/spotlight/blob/main/torch_spotlight/spotlight.py


def gaussian_probs(mean, precision, x):
    # Similarity kernel: describe how similar each point in x is to mean as number in
    # [0, 1]
    # - mean: (dims) vector
    # - precision: (dims, dims) precision matrix; must be PSD
    # - x: (num_points, dims) set of points
    dists = torch.sum(((x - mean) @ precision) * (x - mean), axis=1)
    return torch.exp(-dists / 2)


def md_adversary_weights(mean, precision, x, losses, counts=None):
    # Calculate normalized weights, average loss, and spotlight size for current mean
    # and precision settings
    # - mean, precision, x: as in gaussian_probs
    # - losses: (num_points) vector of losses
    # - counts: (num_points) vector of number of copies of each point to include.
    # defaults to all-ones.

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
    y = losses.clone().to(torch.float).to(device=device)
    dimensions = x.shape[1]

    mean = torch.zeros((dimensions,), requires_grad=True, device=device).to(torch.float)

    log_precision = torch.tensor(np.log(0.0001), requires_grad=True, device=device)
    optimizer = optim.Adam([mean, log_precision], lr=learning_rate)

    scheduler = ResetOnPlateau(
        optimizer, patience=scheduler_patience, factor=scheduler_decay
    )

    num_steps = len(barrier_x_schedule)

    objective_history = []
    total_weight_history = []
    lr_history = []

    for t in range(num_steps):  # removed tqdm here
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
