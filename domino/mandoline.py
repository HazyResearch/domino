"""
Implementation of Mandoline.
Source: Karan Goel
"""

import cytoolz as tz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
import pickle

from scipy.special import logsumexp
from robustnessgym import BinarySentiment
from tqdm.auto import tqdm
from types import SimpleNamespace


def Phi(D, edge_list):
    """Use the edge set of the dependency graph G to construct the representation
    function Phi. (Step 2 of Algorithm 1)
    """
    if edge_list is not None:
        pairwise_terms = (
            D[np.arange(len(D)), edge_list[:, 0][:, np.newaxis]].T
            * D[np.arange(len(D)), edge_list[:, 1][:, np.newaxis]].T
        )
        return np.concatenate([D, pairwise_terms], axis=1)
    else:
        return D


def log_partition_ratio(x, Phi_D_src, n_src):
    return np.log(n_src) - logsumexp(Phi_D_src.dot(x))


def solver(D_src, D_tgt, edge_list):
    """Optimize the objective in eq 4"""
    D_src, D_tgt = np.copy(D_src), np.copy(D_tgt)
    if np.min(D_src) == 0:
        D_src[D_src == 0] = -1
        D_tgt[D_tgt == 0] = -1
    # Edge list encoding dependencies between gs
    if edge_list is not None:
        edge_list = np.array(edge_list)

    # Create the potential matrices
    Phi_D_tgt, Phi_D_src = Phi(D_tgt, edge_list), Phi(D_src, edge_list)

    # Number of examples
    n_src, n_tgt = Phi_D_src.shape[0], Phi_D_tgt.shape[0]

    def f(x):
        obj = Phi_D_tgt.dot(x).sum() - n_tgt * logsumexp(Phi_D_src.dot(x))
        return -obj

    # Solve
    opt = scipy.optimize.minimize(
        f, x0=np.random.randn(Phi_D_tgt.shape[1]), method="BFGS"
    )

    return SimpleNamespace(
        opt=opt,
        Phi_D_src=Phi_D_src,
        Phi_D_tgt=Phi_D_tgt,
        n_src=n_src,
        n_tgt=n_tgt,
        edge_list=edge_list,
    )


def log_density_ratio(D, solved):
    Phi_D = Phi(D, None)
    return Phi_D.dot(solved.opt.x) + log_partition_ratio(
        solved.opt.x, solved.Phi_D_src, solved.n_src
    )


def get_k_most_unbalanced_gs(D_src, D_tgt, k):
    marginal_diff = np.abs(D_src.mean(axis=0) - D_tgt.mean(axis=0))
    differences = np.sort(marginal_diff)[-k:]
    indices = np.argsort(marginal_diff)[-k:]
    return list(indices), list(differences)


def weighted_estimator(weights, empirical_mat):
    if weights is None:
        return np.mean(empirical_mat, axis=0)
    return np.sum(weights[:, np.newaxis] * empirical_mat, axis=0)


def run_estimation_experiment(
    D_src, D_tgt, indices, edge_list, empirical_mat_list_src, empirical_mat_list_tgt
):
    """
    Main entry point to run an experiment with Mandoline.

    D_src: binary source matrix (n_s x d) with 0/1 entries
    D_tgt: binary target matrix (n_t x d) with 0/1 entries
    indices: list of integers used to restrict the procedure to some subset of the columns in D_src, D_tgt
    edge_list: edge pairs (e.g. [(0, 1)]) that indicate correlation in the graphical model structure
    empirical_mat_list_src: list of source matrices for which reweighted means will be computed
    empirical_mat_list_tgt: list of target matrices for which direct means will be computed
    """
    assert len(empirical_mat_list_src) == len(empirical_mat_list_tgt)
    # Run the solver
    solved = solver(
        D_src[:, indices] if indices is not None else D_src,
        D_tgt[:, indices] if indices is not None else D_tgt,
        edge_list,
    )
    # Compute the weights on the source dataset
    density_ratios = np.e ** log_density_ratio(solved.Phi_D_src, solved)
    weights = density_ratios / np.sum(density_ratios)

    all_estimates = []
    for mat_src, mat_tgt in zip(empirical_mat_list_src, empirical_mat_list_tgt):
        # Estimates is a 1-D array of estimates for each mat e.g. each mat can correspond to a model's (n x 1) error matrix
        weighted_estimates = weighted_estimator(weights, mat_src)
        source_estimates = weighted_estimator(
            np.ones(solved.n_src) / solved.n_src, mat_src
        )
        target_estimates = weighted_estimator(
            np.ones(solved.n_tgt) / solved.n_tgt, mat_tgt
        )

        all_estimates.append(
            SimpleNamespace(
                weighted=weighted_estimates,
                source=source_estimates,
                target=target_estimates,
            )
        )

    return SimpleNamespace(
        all_estimates=all_estimates, solved=solved, weights=weights, indices=indices
    )


def diagnostics(results, D_src, D_tgt):
    print(results.all_estimates)
    print(
        "Source | Weighted Source | Target marginals for picked g_is (should match ideally)"
    )
    print(
        weighted_estimator(None, D_src[:, results.indices]),
        weighted_estimator(results.weights, D_src[:, results.indices]),
        weighted_estimator(None, D_tgt[:, results.indices]),
    )
    plt.plot(np.sort(results.weights))
    plt.show()


def effective_sample_size(weights):
    n = weights.shape[0]
    return n * (weights.mean()) ** 2 / (weights ** 2).mean()


def get_correlation_structure(m, top_k=None, thresh=None, min_thresh=0.1, lmbda=0.1):
    # Compute the inverse covariance matrix
    inv_cov = np.linalg.inv(np.dot(m.T, m) / m.shape[0] + np.eye(m.shape[1]) * lmbda)

    # Absolute values
    inv_cov = np.abs(inv_cov)

    # Set diagonal values to 0.
    inv_cov[np.arange(inv_cov.shape[0]), np.arange(inv_cov.shape[0])] = 0.0

    # Create an adjacency matrix
    inv_cov_adj = np.zeros_like(inv_cov)

    if top_k:
        # Indices for top-k values
        top_k_entries = np.unravel_index(
            np.argsort(inv_cov.ravel())[-top_k * 2 :], inv_cov.shape
        )
        inv_cov_adj[top_k_entries] = 1
        inv_cov_adj[inv_cov <= min_thresh] = 0.0
    else:
        inv_cov_adj[inv_cov > thresh] = 1
    inv_cov_adj[np.arange(inv_cov_adj.shape[0]), np.arange(inv_cov_adj.shape[0])] = 0

    # Figure out the edges
    edges = np.array([t for t in zip(*np.where(inv_cov_adj)) if t[0] < t[1]])

    return edges
