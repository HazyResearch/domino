
from typing import Union

import meerkat as mk
import numpy as np
from tqdm import tqdm

from domino.utils import unpack_args

from .abstract import Slicer


class BarlowSlicer(Slicer):

    r"""
    Slice Discovery based on the Barlow.

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

        from domino import BarlowSlicer
        dp = ...  # Load dataset into a Meerkat DataPanel

        # split dataset
        valid_dp = dp.lz[dp["split"] == "valid"]
        test_dp = dp.lz[dp["split"] == "test"]

        domino = BarlowSlicer()
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
        pbar: bool = True,
    ):
        super().__init__(n_slices=n_slices)

       

    def fit(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> BarlowSlicer:
        """
        Fit the decision tree to data. 

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
            Must call ``BarlowSlicer.fit`` prior to calling ``BarlowSlicer.predict``.


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
            Must call ``BarlowSlicer.fit`` prior to calling
            ``BarlowSlicer.predict_proba``.


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
        embeddings, targets, pred_probs = convert_to_numpy(
            embeddings, targets, pred_probs
        )

        if self.pca is not None:
            embeddings = self.pca.transform(X=embeddings)

        clusters = self.mm.predict_proba(embeddings, y=targets, y_hat=pred_probs)

        return clusters[:, self.slice_cluster_indices]




def failure_explanation(imagenet_path, class_name, grouping, model_name = "standard"):
    robust_model_name = 'robust_resnet50.pth'
    robust_model = load_robust_model()
    
    imagenet_subset = ImageNetSubset(imagenet_path, class_name, grouping, model_name)
        
    train_features, train_labels, train_preds = extract_features(robust_model, imagenet_subset)
    
    train_success = (train_preds == train_labels)
    train_failure = np.logical_not(train_success)
    
    train_base_error_rate = np.sum(train_failure)/len(train_failure)
    
    sparse_features, feature_indices = select_important_features(train_features, train_failure, 
                                                                 num_features=50, method='mutual_info')    
    
    decision_tree = train_decision_tree(sparse_features, train_failure, 
                                        max_depth=1, criterion="entropy")
    train_precision, train_recall, train_ALER = decision_tree.compute_precision_recall(
        sparse_features, train_failure)
    
    class_name = class_names[class_index]
    
    
    
    print_with_stars(" Training Data Summary ", prefix="\n")
    print('Grouping by {:s} for class name: {:s}'.format(grouping, class_name))
    print('Number of correctly classified: {:d}'.format(np.sum(train_success)))
    print('Number of incorrectly classified: {:d}'.format(np.sum(train_failure)))
    print('Total size of the dataset: {:d}'.format(len(train_failure)))
    print('Train Base_Error_Rate (BER): {:.4f}\n'.format(train_base_error_rate))

    print_with_stars(" Decision Tree Summary (evaluated on training data) ")
    print('Tree Precision: {:.4f}'.format(train_precision))
    print('Tree Recall: {:.4f}'.format(train_recall))
    print('Tree ALER (ALER of the root node): {:.4f}\n'.format(train_ALER))

    
    error_rate_array, error_coverage_array = decision_tree.compute_leaf_error_rate_coverage(
                                                sparse_features, train_failure)

    important_leaf_ids = important_leaf_nodes(decision_tree, error_rate_array, error_coverage_array)
    for leaf_id in important_leaf_ids[:1]:
        leaf_precision = error_rate_array[leaf_id]
        leaf_recall = error_coverage_array[leaf_id]

        decision_path = decision_tree.compute_decision_path(leaf_id)

        print_with_stars(" Failure statistics for leaf[{:d}] ".format(leaf_id))
        print('Leaf Error_Rate (ER): {:.4f}'.format(leaf_precision))
        print('Leaf Error_Coverage (EC): {:.4f}'.format(leaf_recall))
        print('Leaf Importance_Value (IV): {:.4f}'.format(leaf_precision*leaf_recall))

        
        leaf_failure_indices = decision_tree.compute_leaf_truedata(sparse_features, 
                                                                   train_failure, leaf_id)
        display_failures(leaf_id, leaf_failure_indices, imagenet_subset, grouping, num_images=6)
        
        print_with_stars(" Decision tree path from root to leaf[{:d}] ".format(leaf_id))
        for node in decision_path:
            node_id, feature_id, feature_threshold, direction = node
            
            if direction == 'left':
                print_str = "Feature[{:d}] < {:.6f} (left branching, lower feature activation)".format(
                    feature_id, feature_threshold)
            else:
                print_str = "Feature[{:d}] > {:.6f} (right branching, higher feature activation)".format(
                    feature_id, feature_threshold)

            print(print_str)
        print("")
        
        print_with_stars(" Visualizing features on path from root to leaf[{:d}] ".format(leaf_id))
        print("")
        display_images(decision_path, imagenet_subset, robust_model, train_features, 
                       feature_indices, grouping, num_images=6)