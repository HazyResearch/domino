import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import max_pool2d


def plot_saccade(gaze_seq, s_=100, ax=None):
    gaze_x = 1024 * gaze_seq[:, 1]
    gaze_y = 1024 * gaze_seq[:, 0]
    size = np.ones(len(gaze_x))
    if gaze_seq.shape[1] == 3:
        size = gaze_seq[:, 2]
    if ax:
        ax.scatter(gaze_x, gaze_y, s=s_ * size, alpha=0.7, c="lightgreen")
    else:
        plt.scatter(gaze_x, gaze_y, s=s_ * size, alpha=0.7, c="lightgreen")
    for i in range(len(gaze_seq) - 1):
        # plt.annotate(str(i + 1), (1024*gaze_seq[i,1], 1024*gaze_seq[i,0]))
        if ax:
            ax.arrow(
                gaze_x[i],
                gaze_y[i],
                gaze_x[i + 1] - gaze_x[i],
                gaze_y[i + 1] - gaze_y[i],
            )
        else:
            plt.arrow(
                gaze_x[i],
                gaze_y[i],
                gaze_x[i + 1] - gaze_x[i],
                gaze_y[i + 1] - gaze_y[i],
            )


def generate_ideal_observer(
    starting_loc, alpha, beta, seg_mask, p, threshold=0.99, max_nfix=100, c=(7.5 / 512)
):
    """
    Adapted from: https://github.com/rashidis/bio_based_detectability/blob/main/visual_search.py
    """

    num_points = p * p
    patched_seg_mask = (
        max_pool2d(torch.Tensor(seg_mask).unsqueeze(0), int(1024 / p)).squeeze().numpy()
    )
    target_loc_ndx = patched_seg_mask.reshape(-1).astype(np.bool)

    x_points = np.linspace(1 / (2 * p), 1 - (1 / (2 * p)), p)
    poten_locs = list(itertools.product(x_points, x_points))

    gaze_seq = [starting_loc]

    prior = np.random.uniform(0, 1, num_points)
    posterior = 0
    lsum = np.zeros(num_points)
    deltaHs = np.zeros(num_points)

    k = 0
    while np.max(posterior) < threshold and len(gaze_seq) < max_nfix:
        curr_loc = gaze_seq[-1]
        d_map = calc_dmap(curr_loc, poten_locs, alpha, beta, c)

        w = -0.5 * np.ones(num_points)
        w[target_loc_ndx] = 0.5
        w_map = np.random.normal(w, 1 / (d_map), d_map.shape)

        lsum = lsum + d_map * d_map * w_map
        like = np.exp(lsum)

        posterior = prior * like
        posterior = posterior / sum(posterior)

        for i in range(0, np.shape(poten_locs)[0]):
            d2 = calc_dmap(poten_locs[i], poten_locs, alpha, beta, c)
            deltaHs[i] = sum(d2 * d2 * posterior)

        best_loc_ndx = np.argmax(deltaHs)
        gaze_seq.append(poten_locs[best_loc_ndx])

    return gaze_seq


def calc_dmap(curr_loc, poten_locs, alpha, beta, c):
    d_map = []
    for loc in poten_locs:
        loc_deg = c * np.linalg.norm(np.array(curr_loc) - np.array(loc))
        d_map.append(alpha * np.exp(-beta * loc_deg))

    return np.array(d_map)
