import warnings
from dataclasses import dataclass
from functools import wraps

import meerkat as mk
import numpy as np
import sklearn.cluster as cluster
import torch.nn as nn
from scipy import linalg
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.mixture._base import _check_X, check_random_state
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _estimate_gaussian_covariances_diag,
    _estimate_gaussian_covariances_full,
    _estimate_gaussian_covariances_spherical,
    _estimate_gaussian_covariances_tied,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from domino.sdm.abstract import SliceDiscoveryMethod
from domino.utils import VariableColumn, requires_columns


class MixtureModelSDM(SliceDiscoveryMethod):
    @dataclass
    class Config(SliceDiscoveryMethod.Config):
        weight_y_log_likelihood: float = 1
        covariance_type: str = "diag"
        pca_components: int = 128
        n_clusters: int = 25
        explain_w_model: bool = False
        init_params: str = "error"

    RESOURCES_REQUIRED = {"cpu": 1}

    def __init__(self, config: dict = None, **kwargs):
        super().__init__(config, **kwargs)
        if self.config.pca_components is None:
            self.pca = None
        else:
            self.pca = PCA(n_components=self.config.pca_components)
        self.gmm = ErrorMixtureModel(
            n_components=self.config.n_clusters,
            weight_y_log_likelihood=self.config.weight_y_log_likelihood,
            covariance_type=self.config.covariance_type,
            init_params=self.config.init_params,
        )

    @requires_columns(dp_arg="data_dp", columns=[VariableColumn("self.config.emb")])
    def fit(
        self,
        data_dp: mk.DataPanel,
        model: nn.Module = None,
    ):
        emb = data_dp[self.config.emb].data
        if self.pca is not None:
            self.pca.fit(X=emb[:1000])
            emb = self.pca.transform(X=emb)
        self.gmm.fit(X=emb, y=data_dp["target"], y_hat=data_dp["pred"])

        self.slice_cluster_indices = (
            -np.abs((self.gmm.y_probs[:, 1] - self.gmm.y_hat_probs[:, 1]))
        ).argsort()[: self.config.n_slices]
        return self

    @requires_columns(dp_arg="data_dp", columns=[VariableColumn("self.config.emb")])
    def transform(
        self,
        data_dp: mk.DataPanel,
    ):
        emb = data_dp[self.config.emb].data
        if self.pca is not None:
            emb = self.pca.transform(X=emb)
        dp = data_dp.view()
        clusters = self.gmm.predict_proba(
            emb, y=data_dp["target"], y_hat=data_dp["pred"]
        )

        dp["pred_slices"] = clusters[:, self.slice_cluster_indices]
        return dp

    # @requires_columns(dp_arg="words_dp", columns=[VariableColumn("self.config.emb")])
    def explain(
        self, words_dp: mk.DataPanel, data_dp: mk.DataPanel = None
    ) -> mk.DataPanel:
        if not self.config.explain_w_model:
            return super().explain(words_dp, data_dp)
        words_dp = words_dp.view()
        emb = words_dp["emb"]
        if self.pca is not None:
            emb = self.pca.transform(X=emb)
        words_dp["pred_slices"] = self.gmm.predict_proba(X=emb)[
            :, self.slice_cluster_indices
        ]
        return words_dp[["word", "pred_slices", "frequency"]]


class ErrorMixtureModel(GaussianMixture):
    @wraps(GaussianMixture.__init__)
    def __init__(self, *args, weight_y_log_likelihood: float = 1, **kwargs):
        self.weight_y_log_likelihood = weight_y_log_likelihood
        super().__init__(*args, **kwargs)

    def _initialize_parameters(self, X, y, y_hat, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_samples, _ = X.shape

        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components))
            label = (
                cluster.KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X)
                .labels_
            )
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        elif self.init_params == "error":
            num_classes = y.shape[-1]
            if self.n_components < num_classes ** 2:
                raise ValueError(
                    "Can't use parameter init 'error' when `n_components` < `num_classes **2`"
                )
            resp = np.matmul(y[:, :, np.newaxis], y_hat[:, np.newaxis, :]).reshape(
                len(y), -1
            )
            resp = np.concatenate(
                [resp]
                * (
                    int(self.n_components / (num_classes ** 2))
                    + (self.n_components % (num_classes ** 2) > 0)
                ),
                axis=1,
            )[:, : self.n_components]
            resp /= resp.sum(axis=1)[:, np.newaxis]

            resp += random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]

        else:
            raise ValueError(
                "Unimplemented initialization method '%s'" % self.init_params
            )

        self._initialize(X, y, y_hat, resp)

    def _initialize(self, X, y, y_hat, resp):
        """Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances, y_probs, y_hat_probs = _estimate_parameters(
            X, y, y_hat, resp, self.reg_covar, self.covariance_type
        )
        weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init
        self.y_probs, self.y_hat_probs = y_probs, y_hat_probs
        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type
            )
        elif self.covariance_type == "full":
            self.precisions_cholesky_ = np.array(
                [
                    linalg.cholesky(prec_init, lower=True)
                    for prec_init in self.precisions_init
                ]
            )
        elif self.covariance_type == "tied":
            self.precisions_cholesky_ = linalg.cholesky(
                self.precisions_init, lower=True
            )
        else:
            self.precisions_cholesky_ = self.precisions_init

    def fit(self, X, y, y_hat):

        self.fit_predict(X, y, y_hat)
        return self

    def _preprocess_ys(self, y: np.ndarray = None, y_hat: np.ndarray = None):
        if y is not None:
            y = label_binarize(y, classes=np.arange(np.max(y) + 1))
            if y.shape[-1] == 1:
                # binary targets transform to a column vector with label_binarize
                y = np.array([1 - y[:, 0], y[:, 0]]).T
        if y is not None:
            if len(y_hat.shape) == 1:
                y_hat = np.array([1 - y_hat, y_hat]).T
        return y, y_hat

    def fit_predict(self, X, y, y_hat):
        y, y_hat = self._preprocess_ys(y, y_hat)

        X = _check_X(X, self.n_components, ensure_min_samples=2)
        self._check_n_features(X, reset=True)
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, y, y_hat, random_state)

            lower_bound = -np.infty if do_init else self.lower_bound_

            for n_iter in tqdm(range(1, self.max_iter + 1)):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X, y, y_hat)
                self._m_step(X, y, y_hat, log_resp)
                lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)
                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X, y, y_hat)

        return log_resp.argmax(axis=1)

    def predict_proba(
        self, X: np.ndarray, y: np.ndarray = None, y_hat: np.ndarray = None
    ):
        y, y_hat = self._preprocess_ys(y, y_hat)

        check_is_fitted(self)
        X = _check_X(X, None, self.means_.shape[1])
        _, log_resp = self._estimate_log_prob_resp(X, y, y_hat)
        return np.exp(log_resp)

    def _m_step(self, X, y, y_hat, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        resp = np.exp(log_resp)
        n_samples, _ = X.shape
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.y_probs,
            self.y_hat_probs,
        ) = _estimate_parameters(
            X, y, y_hat, resp, self.reg_covar, self.covariance_type
        )
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def _e_step(self, X, y, y_hat):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X, y, y_hat)
        return np.mean(log_prob_norm), log_resp

    def _estimate_log_prob_resp(self, X, y=None, y_hat=None):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X, y, y_hat)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _estimate_weighted_log_prob(self, X, y=None, y_hat=None):
        log_prob = self._estimate_log_prob(X) + self._estimate_log_weights()

        if y is not None:
            log_prob += self._estimate_y_log_prob(y) * self.weight_y_log_likelihood

        if y_hat is not None:
            log_prob += (
                self._estimate_y_hat_log_prob(y_hat) * self.weight_y_log_likelihood
            )

        return log_prob

    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
            self.covariances_,
            self.y_probs,
            self.y_hat_probs,
            self.precisions_cholesky_,
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.y_probs,
            self.y_hat_probs,
            self.precisions_cholesky_,
        ) = params

        # Attributes computation
        _, n_features = self.means_.shape

        if self.covariance_type == "full":
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == "tied":
            self.precisions_ = np.dot(
                self.precisions_cholesky_, self.precisions_cholesky_.T
            )
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        return super()._n_parameters() + 2 * self.n_components

    def _estimate_y_log_prob(self, y):
        """Estimate the Gaussian distribution parameters.

        Parameters
        ----------
        y: array-like of shape (n_samples, n_classes)

        y_hat: array-like of shpae (n_samples, n_classes)
        """
        # add epsilon to avoid "RuntimeWarning: divide by zero encountered in log"
        return np.log(np.dot(y, self.y_probs.T) + np.finfo(self.y_probs.dtype).eps)

    def _estimate_y_hat_log_prob(self, y_hat):
        """Estimate the Gaussian distribution parameters.

        Parameters
        ----------
        y: array-like of shape (n_samples, n_classes)

        y_hat: array-like of shpae (n_samples, n_classes)
        """
        # add epsilon to avoid "RuntimeWarning: divide by zero encountered in log"
        return np.log(
            np.dot(y_hat, self.y_hat_probs.T) + np.finfo(self.y_hat_probs.dtype).eps
        )


def _estimate_parameters(X, y, y_hat, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    y: array-like of shape (n_samples, n_classes)

    y_hat: array-like of shpae (n_samples, n_classes)

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # (n_components, )
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)

    y_probs = np.dot(resp.T, y) / nk[:, np.newaxis]  # (n_components, n_classes)
    y_hat_probs = np.dot(resp.T, y_hat) / nk[:, np.newaxis]  # (n_components, n_classes)

    return nk, means, covariances, y_probs, y_hat_probs
