from __future__ import annotations

import warnings
from functools import wraps
from typing import Union

import meerkat as mk
import numpy as np
import sklearn.cluster as cluster
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
from tqdm.auto import tqdm

from domino.utils import unpack_args

from .abstract import Slicer


class DominoSlicer(Slicer):

    r"""
    Slice Discovery based on the Domino Mixture Model.

    Discover slices by jointly modeling a mixture of input embeddings (e.g. activations
    from a trained model), class labels, and model predictions. This encourages slices
    that are homogeneous with respect to error type (e.g. all false positives).

    Examples
    --------
    Suppose you've trained a model and stored its predictions on a dataset in
    a `Meerkat DataPanel <https://github.com/robustness-gym/meerkat>`_ with columns
    "emb", "target", and "pred_probs". After loading the DataPanel, you can discover
    underperforming slices of the validation dataset with the following:

    .. code-block:: python

        from domino import DominoSlicer
        dp = ...  # Load dataset into a Meerkat DataPanel

        # split dataset
        valid_dp = dp.lz[dp["split"] == "valid"]
        test_dp = dp.lz[dp["split"] == "test"]

        domino = DominoSlicer()
        domino.fit(
            data=valid_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )
        dp["domino_slices"] = domino.transform(
            data=test_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )


    Args:
        n_slices (int, optional): The number of slices to discover.
            Defaults to 5.
        covariance_type (str, optional): The type of covariance parameter
            :math:`\mathbf{\Sigma}` to use. Same as in sklearn.mixture.GaussianMixture.
            Defaults to "diag", which is recommended.
        n_pca_components (Union[int, None], optional): The number of PCA components
            to use. If ``None``, then no PCA is performed. Defaults to 128.
        n_mixture_components (int, optional): The number of clusters in the mixture
            model, :math:`\bar{k}`. This differs from ``n_slices`` in that the
            ``DominoSDM`` only returns the top ``n_slices`` with the highest error rate
            of the ``n_mixture_components``. Defaults to 25.
        y_log_likelihood_weight (float, optional): The weight :math:`\gamma` applied to
            the :math:`P(Y=y_{i} | S=s)` term in the log likelihood during the E-step.
            Defaults to 1.
        y_hat_log_likelihood_weight (float, optional): The weight :math:`\hat{\gamma}`
            applied to the :math:`P(\hat{Y} = h_\theta(x_i) | S=s)` term in the log
            likelihood during the E-step. Defaults to 1.
        max_iter (int, optional): The maximum number of iterations to run. Defaults
            to 100.
        init_params (str, optional): The initialization method to use. Options are
            the same as in sklearn.mixture.GaussianMixture plus one addition,
            "confusion". If "confusion",  the clusters are initialized such that almost
            all of the examples in a cluster come from same cell in the confusion
            matrix. See Notes below for more details. Defaults to "confusion".
        confusion_noise (float, optional): Only used if ``init_params="confusion"``.
            The scale of noise added to the confusion matrix initialization. See notes
            below for more details.
            Defaults to 0.001.
        random_state (Union[int, None], optional): The random seed to use when
            initializing  the parameters.

    Notes
    -----

    The mixture model is an extension of a standard Gaussian Mixture Model. The model is
    based on the assumption that data is generated according to the following generative
    process.

    * Each example belongs to one of :math:`\bar{k}` slices. This slice
      :math:`S` is sampled from a categorical
      distribution :math:`S \sim Cat(\mathbf{p}_S)` with parameter :math:`\mathbf{p}_S
      \in\{\mathbf{p} \in \mathbb{R}_+^{\bar{k}} : \sum_{i = 1}^{\bar{k}} p_i = 1\}`
      (see ``DominoSDM.mm.weights_``).
    * Given the slice :math:`S'`, the embeddings are normally distributed
      :math:`Z | S \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma}`)  with parameters
      mean :math:`\mathbf{\mu} \in \mathbb{R}^d` (see ``DominoSDM.mm.means_``) and
      :math:`\mathbf{\Sigma} \in \mathbb{S}^{d}_{++}`
      (see ``DominoSDM.mm.covariances_``;
      normally this parameter is constrained to the set of symmetric positive definite
      :math:`d \\times d` matrices, however the argument ``covariance_type`` allows for
      other constraints).
    * Given the slice, the labels vary as a categorical
      :math:`Y |S \sim Cat(\mathbf{p})` with parameter :math:`\mathbf{p}
      \in \{\mathbf{p} \in \mathbb{R}^c_+ : \sum_{i = 1}^c p_i = 1\}` (see
      ``DominoSDM.mm.y_probs``).
    * Given the slice, the model predictions also vary as a categorical
      :math:`\hat{Y} | S \sim Cat(\mathbf{\hat{p}})` with parameter
      :math:`\mathbf{\hat{p}} \in \{\mathbf{\hat{p}} \in \mathbb{R}^c_+ :
      \sum_{i = 1}^c \hat{p}_i = 1\}` (see ``DominoSDM.mm.y_hat_probs``).

    The mixture model is, thus, parameterized by :math:`\phi = [\mathbf{p}_S, \mu,
    \Sigma, \mathbf{p}, \mathbf{\hat{p}}]` corresponding to the attributes
    ``weights_, means_, covariances_, y_probs, y_hat_probs`` respectively. The
    log-likelihood over the :math:`n` examples in the validation dataset :math:`D_v` is
    given as followsand maximized using expectation-maximization:

    .. math::
        \ell(\phi) = \sum_{i=1}^n \log \sum_{s=1}^{\hat{k}} P(S=s)P(Z=z_i| S=s)
        P( Y=y_i| S=s)P(\hat{Y} = h_\theta(x_i) | S=s)

    We include two optional hyperparameters
    :math:`\gamma, \hat{\gamma} \in \mathbb{R}_+`
    (see ``y_log_liklihood_weight`` and ``y_hat_log_likelihood_weight`` below) that
    balance the importance of modeling the class labels and predictions against the
    importance of modeling the embedding. The modified log-likelihood over :math:`n`
    examples is given as follows:

    .. math::
        \ell(\phi) = \sum_{i=1}^n \log \sum_{s=1}^{\hat{k}} P(S=s)P(Z=z_i| S=s)
        P( Y=y_i| S=s)^\gamma P(\hat{Y} = h_\theta(x_i) | S=s)^{\hat{\gamma}}

    .. attention::
        Although we model the prediction :math:`\hat{Y}` as a categorical random
        variable, in practice predictions are sometimes "soft" (e.g. the output
        of a softmax layer is a probability distribution over labels, not a single
        label). In these cases, the prediction :math:`\hat{Y}` is technically a
        dirichlet random variable (i.e. a distribution over distributions).

        However, to keep the implementation simple while still leveraging the extra
        information provided by "soft" predictions, we naïvely plug the "soft"
        predictions directly into the categorical PMF in the E-step and the update in
        the M-step. Specifically, during the E-step, instead of computing the
        categorical PMF :math:`P(\hat{Y}=\hat{y_i} | S=s)` we compute
        :math:`\sum_{j=1}^c \hat{y_i}(j) P(\hat{Y}=j | S=s)` where :math:`\hat{y_i}(j)`
        is the "soft" prediction for class :math:`j` (we can
        think of this like we're marginalizing out the uncertainty in the prediction).
        During the M-step, we compute a "soft" update for the categorical parameters
        :math:`p_j^{(s)} = \sum_{i=1}^n Q(s,i) \hat{y_i}(j)` where :math:`Q(s,i)`
        is the "responsibility" of slice :math:`s` towards the data point :math:`i`.

    When using ``"confusion"`` initialization, each slice $s^{(j)}$ is assigned a
    :math:`y^{(j)}\in \mathcal{Y}` and :math:`\hat{y}^{(j)} \in \mathcal{Y}` (*i.e.*
    each slice is assigned a cell in the confusion matrix). This is typically done in a
    round-robin fashion so that there are at least
    :math:`\floor{\hat{k} / {|\mathcal{Y}|^2}}`
    slices assigned to each cell in the confusion matrix. Then, we fill in the initial
    responsibility matrix :math:`Q \in \mathbb{R}^{n \times \hat{k}}`, where each cell
    :math:`Q_{ij}` corresponds to our model's initial estimate of
    :math:`P(S=s^{(j)}|Y=y_i,
    \hat{Y}=\hat{y}_i)`. We do this according to

    .. math::
        \bar{Q}_{ij} \leftarrow
        \begin{cases}
            1 + \epsilon & y_i=y^{(j)} \land \hat{y}_i = \hat{y}^{(j)} \\
            \epsilon & \text{otherwise}
        \end{cases}

    .. math::
        Q_{ij} \leftarrow \frac{\bar{Q}_{ij} } {\sum_{l=1}^{\hat{k}} \bar{Q}_{il}}

    where :math:`\epsilon` is random noise which ensures that slices assigned to the
    same confusion matrix cell won't have the exact same initialization. We sample
    :math:`\epsilon` uniformly from the range ``(0, confusion_noise]``.

    """

    def __init__(
        self,
        n_slices: int = 5,
        covariance_type: str = "diag",
        n_pca_components: Union[int, None] = 128,
        n_mixture_components: int = 25,
        y_log_likelihood_weight: float = 1,
        y_hat_log_likelihood_weight: float = 1,
        max_iter: int = 100,
        init_params: str = "confusion",
        confusion_noise: float = 1e-3,
        random_state: int = None,
    ):
        super().__init__(n_slices=n_slices)

        self.config.covariance_type = covariance_type
        self.config.n_pca_components = n_pca_components
        self.config.n_mixture_components = n_mixture_components
        self.config.init_params = init_params
        self.config.confusion_noise = confusion_noise
        self.config.y_log_likelihood_weight = y_log_likelihood_weight
        self.config.y_hat_log_likelihood_weight = y_hat_log_likelihood_weight
        self.config.max_iter = max_iter

        if self.config.n_pca_components is None:
            self.pca = None
        else:
            self.pca = PCA(n_components=self.config.n_pca_components)

        self.mm = DominoMixture(
            n_components=self.config.n_mixture_components,
            y_log_likelihood_weight=self.config.y_log_likelihood_weight,
            y_hat_log_likelihood_weight=self.config.y_hat_log_likelihood_weight,
            covariance_type=self.config.covariance_type,
            init_params=self.config.init_params,
            max_iter=self.config.max_iter,
            confusion_noise=self.config.confusion_noise,
            random_state=random_state,
        )

    def fit(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> DominoSlicer:
        """
        Fit the mixture model to data.

        Args:
            data (mk.DataPanel, optional): A `Meerkat DataPanel` with columns for
                embeddings, targets, and prediction probabilities. The names of the
                columns can be specified with the ``embeddings``, ``targets``, and
                ``pred_probs`` arguments. Defaults to None.
            embeddings (Union[str, np.ndarray], optional): The name of a colum in
                ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
                of shape (n_samples, dimension of embedding). Defaults to
                "embedding".
            targets (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding class labels. If ``data`` is ``None``, then an
                np.ndarray of shape (n_samples,). Defaults to "target".
            pred_probs (Union[str, np.ndarray], optional): The name of
                a column in ``data`` holding model predictions (can either be "soft"
                probability scores or "hard" 1-hot encoded predictions). If
                ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
                or (n_samples,) in the binary case. Defaults to "pred_probs".

        Returns:
            DominoSDM: Returns a fit instance of DominoSDM.
        """
        embeddings, targets, pred_probs = unpack_args(
            data, embeddings, targets, pred_probs
        )

        if self.pca is not None:
            self.pca.fit(X=embeddings)
            embeddings = self.pca.transform(X=embeddings)

        self.mm.fit(X=embeddings, y=targets, y_hat=pred_probs)

        self.slice_cluster_indices = (
            -np.abs((self.mm.y_hat_probs - self.mm.y_probs).max(axis=1))
        ).argsort()[: self.config.n_slices]
        return self

    def predict(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> np.ndarray:
        """
        Get probabilistic slice membership for data using a fit mixture model.


        .. caution::
            Must call ``DominoSlicer.fit`` prior to calling ``DominoSlicer.predict``.


        Args:
            data (mk.DataPanel, optional): A `Meerkat DataPanel` with columns for
                embeddings, targets, and prediction probabilities. The names of the
                columns can be specified with the ``embeddings``, ``targets``, and
                ``pred_probs`` arguments. Defaults to None.
            embeddings (Union[str, np.ndarray], optional): The name of a colum in
                ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
                of shape (n_samples, dimension of embedding). Defaults to
                "embedding".
            targets (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding class labels. If ``data`` is ``None``, then an
                np.ndarray of shape (n_samples,). Defaults to "target".
            pred_probs (Union[str, np.ndarray], optional): The name of
                a column in ``data`` holding model predictions (can either be "soft"
                probability scores or "hard" 1-hot encoded predictions). If
                ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
                or (n_samples,) in the binary case. Defaults to "pred_probs".

        Returns:
            np.ndarray: A binary ``np.ndarray`` of shape (n_samples, n_slices) where
                values are either 1 or 0.
        """
        probs = self.predict_proba(
            data=data,
            embeddings=embeddings,
            targets=targets,
            pred_probs=pred_probs,
        )
        preds = np.zeros_like(probs, dtype=np.int32)
        preds[np.arange(preds.shape[0]), probs.argmax(axis=-1)] = 1
        return preds

    def predict_proba(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> np.ndarray:
        """
        Get probabilistic slice membership for data using a fit mixture model.

        .. caution::
            Must call ``DominoSlicer.fit`` prior to calling
            ``DominoSlicer.predict_proba``.


        Args:
            data (mk.DataPanel, optional): A `Meerkat DataPanel` with columns for
                embeddings, targets, and prediction probabilities. The names of the
                columns can be specified with the ``embeddings``, ``targets``, and
                ``pred_probs`` arguments. Defaults to None.
            embeddings (Union[str, np.ndarray], optional): The name of a colum in
                ``data`` holding embeddings. If ``data`` is ``None``, then an np.ndarray
                of shape (n_samples, dimension of embedding). Defaults to
                "embedding".
            targets (Union[str, np.ndarray], optional): The name of a column in
                ``data`` holding class labels. If ``data`` is ``None``, then an
                np.ndarray of shape (n_samples,). Defaults to "target".
            pred_probs (Union[str, np.ndarray], optional): The name of
                a column in ``data`` holding model predictions (can either be "soft"
                probability scores or "hard" 1-hot encoded predictions). If
                ``data`` is ``None``, then an np.ndarray of shape (n_samples, n_classes)
                or (n_samples,) in the binary case. Defaults to "pred_probs".

        Returns:
            np.ndarray: A ``np.ndarray`` of shape (n_samples, n_slices) where values in
                are in range [0,1] and rows sum to 1.
        """
        embeddings, targets, pred_probs = unpack_args(
            data, embeddings, targets, pred_probs
        )

        if self.pca is not None:
            embeddings = self.pca.transform(X=embeddings)

        clusters = self.mm.predict_proba(embeddings, y=targets, y_hat=pred_probs)

        return clusters[:, self.slice_cluster_indices]


class DominoMixture(GaussianMixture):
    @wraps(GaussianMixture.__init__)
    def __init__(
        self,
        *args,
        y_log_likelihood_weight: float = 1,
        y_hat_log_likelihood_weight: float = 1,
        confusion_noise: float = 1e-3,
        **kwargs,
    ):
        self.y_log_likelihood_weight = y_log_likelihood_weight
        self.y_hat_log_likelihood_weight = y_hat_log_likelihood_weight
        self.confusion_noise = confusion_noise
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
        elif self.init_params == "confusion":
            num_classes = y.shape[-1]
            if self.n_components < num_classes ** 2:
                raise ValueError(
                    "Can't use parameter init 'error' when "
                    "`n_components` < `num_classes **2`"
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

            resp += (
                random_state.rand(n_samples, self.n_components) * self.confusion_noise
            )
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
        if y_hat is not None:
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

            for n_iter in tqdm(range(1, self.max_iter + 1), colour="#f17a4a"):
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
            log_prob += self._estimate_y_log_prob(y) * self.y_log_likelihood_weight

        if y_hat is not None:
            log_prob += (
                self._estimate_y_hat_log_prob(y_hat) * self.y_hat_log_likelihood_weight
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
