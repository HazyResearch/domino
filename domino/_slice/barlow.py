from __future__ import annotations
from collections import defaultdict
from multiprocessing.sharedctypes import Value
from typing import Union

import meerkat as mk
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from domino.utils import convert_to_numpy, unpack_args
from .abstract import Slicer


class BarlowSlicer(Slicer):

    r"""
    Slice Discovery based on the Barlow [singla_2021]_.

    Discover slices using a decision tree. TODO(singlasahil14): add any more details 
    describing your method  
    
    .. note: 

        The authors of the Barlow paper use this slicer with embeddings from a  
        classifier trained using an adversarially robust loss [engstrom_2019]_. 
        To compute embeddings using such a classifier, pass ``encoder="robust"`` to
        ``domino.embed``.

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

        barlow = BarlowSlicer()
        barlow.fit(
            data=valid_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )
        dp["barlow_slices"] = barlow.transform(
            data=test_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )


    Args:
        n_slices (int, optional): The number of slices to discover.
            Defaults to 5.
        max_depth (str, optional): The maximum depth of the desicion tree. Defaults to
            3. If None, then nodes are expanded until all leaves are pure or until all
            leaves contain less than 2 samples. See SKlearn documentation for more
            information.
        n_features (int, optional): The number features from the embedding
            to use. Defaults to 128. Features are selcted using mutual information
            estimate.
        pbar (bool, optional): Whether to show a progress bar. Ignored for barlow.


    .. [singla_2021]

        Singla, Sahil, et al. "Understanding failures of deep networks via robust
        feature extraction." Proceedings of the IEEE/CVF Conference on Computer Vision
        and Pattern Recognition. 2021.


    .. [engstrom_2019]

       @misc{robustness,
            title={Robustness (Python Library)},
            author={Logan Engstrom and Andrew Ilyas and Hadi Salman and Shibani 
            Santurkar and Dimitris Tsipras},
            year={2019},
            url={https://github.com/MadryLab/robustness}
        }

    """

    def __init__(
        self,
        n_slices: int = 5,
        max_depth: int = 3,  # TODO(singlasahil14): confirm this default
        n_features: int = 128,  # TODO(singlasahil14): confirm this default
        pbar: bool = True,
    ):
        super().__init__(n_slices=n_slices)
        self.config.max_depth = max_depth
        self.config.n_features = n_features

        # parameters set after a call to fit
        self._feature_indices = None 
        self._important_leaf_ids = None 
        self._decision_tree = None 

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
            BarlowSlicer: Returns a fit instance of BarlowSlicer.
        """
        embeddings, targets, pred_probs = unpack_args(
            data, embeddings, targets, pred_probs
        )
        embeddings, targets, pred_probs = convert_to_numpy(
            embeddings, targets, pred_probs
        )

        if pred_probs.ndim > 1:
            preds = pred_probs.argmax(axis=-1)
        else:
            preds = pred_probs > 0.5

        success = preds == targets
        failure = np.logical_not(success)

        sparse_features, feature_indices = _select_important_features(
            embeddings,
            failure,
            num_features=self.config.n_features,
            method="mutual_info",
        )
        self._feature_indices = feature_indices

        decision_tree = _train_decision_tree(
            sparse_features,
            failure,
            max_depth=self.config.max_depth,
            criterion="entropy",
        )

        (
            error_rate_array,
            error_coverage_array,
        ) = decision_tree.compute_leaf_error_rate_coverage(sparse_features, failure)

        important_leaf_ids = _important_leaf_nodes(
            decision_tree, error_rate_array, error_coverage_array
        )

        self._decision_tree = decision_tree
        self._important_leaf_ids = important_leaf_ids

        return self

    def predict(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> np.ndarray:
        """
        Predict slice membership according to the learnt decision tree. 

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
        if self._decision_tree is None:
            raise ValueError(
                "Must call `fit` prior to calling `predict` or `predict_proba`."
            )
        (embeddings,) = unpack_args(data, embeddings)
        (embeddings,) = convert_to_numpy(embeddings)

        embeddings = embeddings[:, self._feature_indices]

        leaves = self._decision_tree.apply(embeddings)  # (n_samples,)

        # convert to 1-hot encoding of size (n_samples, n_slices) using broadcasting
        slices = (
            leaves[:, np.newaxis] == self._important_leaf_ids[np.newaxis, :]
        ).astype(int)
        return slices

    def predict_proba(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
    ) -> np.ndarray:
        """
        Predict slice membership according to the learnt decision tree. 

        .. warning::
            Because the decision tree does not produce probabilistic leaf assignments, 
            this method is equivalent to `predict` 

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
        return self.predict(data, embeddings, targets, pred_probs)


def _mutual_info_select(train_features_class, train_failure_class, num_features=20):
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(train_features_class, train_failure_class, random_state=0)
    important_features_indices = np.argsort(mi)[-num_features:]
    important_features_values = mi[important_features_indices]
    return important_features_indices, important_features_values


def _feature_importance_select(train_features_class, num_features=20):
    fi = np.mean(train_features_class, axis=0)
    important_features_indices = np.argsort(fi)[-num_features:]
    important_features_values = fi[important_features_indices]
    return important_features_indices, important_features_values


def _select_important_features(
    train_features, train_failure, num_features=20, method="mutual_info"
):
    """Perform feature selection using some prespecified method such as
    mutual information.

    Args:
        train_features (_type_): _description_
        train_failure (_type_): _description_
        num_features (int, optional): _description_. Defaults to 20.
        method (str, optional): _description_. Defaults to 'mutual_info'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if method == "mutual_info":
        important_indices, _ = _mutual_info_select(
            train_features, train_failure, num_features=num_features
        )
    elif method == "feature_importance":
        important_indices, _ = _feature_importance_select(
            train_features, num_features=num_features
        )
    else:
        raise ValueError("Unknown feature selection method")

    train_sparse_features = train_features[:, important_indices]
    return train_sparse_features, important_indices


class BarlowDecisionTreeClassifier(DecisionTreeClassifier):
    """Extension of scikit-learn's DecisionTreeClassifier"""

    def fit_tree(self, train_data, train_labels):
        """Learn decision tree using features 'train_data' and labels 'train_labels"""
        num_true = np.sum(train_labels)
        num_false = np.sum(np.logical_not(train_labels))
        if self.class_weight == "balanced":
            self.float_class_weight = num_false / num_true
        elif isinstance(self.class_weight, dict):
            keys_list = list(self.class_weight.keys())
            assert len(keys_list) == 2
            assert 0 in keys_list
            assert 1 in keys_list
            self.float_class_weight = self.class_weight[1]

        self.fit(train_data, train_labels)
        true_dict, false_dict = self.compute_TF_dict(train_data, train_labels)
        self.train_true_dict = dict(true_dict)
        self.train_false_dict = dict(false_dict)

        self._compute_parent()

        true_array = np.array(list(true_dict))
        false_array = np.array(list(false_dict))
        unique_leaf_ids = np.union1d(true_array, false_array)
        self.leaf_ids = unique_leaf_ids

        true_leaves = []

        for leaf_id in unique_leaf_ids:
            true_count = true_dict[leaf_id]
            false_count = false_dict[leaf_id]
            if true_count * self.float_class_weight > false_count:
                true_leaves.append(leaf_id)
        self.true_leaves = true_leaves
        return self

    def _compute_parent(self):
        """Find the parent of every leaf node"""
        n_nodes = self.tree_.node_count
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right

        self.parent = np.zeros(shape=n_nodes, dtype=np.int64)
        stack = [0]
        while len(stack) > 0:
            node_id = stack.pop()

            child_left = children_left[node_id]
            child_right = children_right[node_id]
            if child_left != child_right:
                self.parent[child_left] = node_id
                self.parent[child_right] = node_id
                stack.append(child_left)
                stack.append(child_right)

    def compute_leaf_data(self, data, leaf_id):
        """Find which of the data points lands in the leaf node with identifier 'leaf_id'"""
        leaf_ids = self.apply(data)
        return np.nonzero(leaf_ids == leaf_id)[0]

    def compute_leaf_truedata(self, data, labels, leaf_id):
        """Find which of the data points lands in the leaf node with identifier
        'leaf_id' and for which the prediction is 'true'."""
        leaf_ids = self.apply(data)
        leaf_data_indices = np.nonzero(leaf_ids == leaf_id)[0]
        leaf_failure_labels = labels[leaf_data_indices]
        leaf_failure_indices = leaf_data_indices[leaf_failure_labels]
        return leaf_failure_indices

    def compute_TF_dict(self, data, labels):
        """
        Returns two dictionaries 'true_dict' and 'false_dict'.
        true_dict maps every leaf_id to the number of correctly classified
        data points in the leaf with that leaf_id.
        false_dict maps every leaf_id to the number of incorrectly classified
        data points in the leaf with that leaf_id.
        """

        def create_dict(unique, counts, dtype=int):
            count_dict = defaultdict(dtype)
            for u, c in zip(unique, counts):
                count_dict[u] = count_dict[u] + c
            return count_dict

        leaf_ids = self.apply(data)
        true_leaf_ids = leaf_ids[np.nonzero(labels)]
        false_leaf_ids = leaf_ids[np.nonzero(np.logical_not(labels))]

        true_unique, _, true_unique_counts = np.unique(
            true_leaf_ids, return_index=True, return_counts=True
        )
        true_dict = create_dict(true_unique, true_unique_counts)
        false_unique, _, false_unique_counts = np.unique(
            false_leaf_ids, return_index=True, return_counts=True
        )
        false_dict = create_dict(false_unique, false_unique_counts)
        return true_dict, false_dict

    def compute_precision_recall(self, data, labels, compute_ALER=True):
        """
        Compute precision and recall for the tree. Also compute
        Average Leaf Error Rate if compute_ALER is True.
        """
        true_dict, false_dict = self.compute_TF_dict(data, labels)
        total_true = np.sum(labels)
        total_pred = 0
        total = 0
        for leaf_id in self.true_leaves:
            true_count = true_dict[leaf_id]
            false_count = false_dict[leaf_id]

            total_pred += true_count
            total += true_count + false_count

        precision = total_pred / total
        recall = total_pred / total_true

        if compute_ALER:
            average_precision = self.compute_average_leaf_error_rate(data, labels)
            return precision, recall, average_precision
        else:
            return precision, recall

    def compute_average_leaf_error_rate(self, data, labels):
        """Compute Average Leaf Error Rate using the trained decision tree"""
        num_true = np.sum(labels)
        true_dict, false_dict = self.compute_TF_dict(data, labels)

        avg_leaf_error_rate = 0
        for leaf_id in self.leaf_ids:
            true_count = true_dict[leaf_id]
            false_count = false_dict[leaf_id]
            if true_count + false_count > 0:
                curr_error_coverage = true_count / num_true
                curr_error_rate = true_count / (true_count + false_count)

                avg_leaf_error_rate += curr_error_coverage * curr_error_rate
        return avg_leaf_error_rate

    def compute_decision_path(self, leaf_id, important_features_indices=None):
        """Compute decision_path (the set of decisions used to arrive at a certain
        leaf)
        """
        assert leaf_id in self.leaf_ids

        features_arr = self.tree_.feature
        thresholds_arr = self.tree_.threshold

        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        path = []
        curr_node = leaf_id
        while curr_node > 0:
            parent_node = self.parent[curr_node]

            is_left_child = children_left[parent_node] == curr_node
            is_right_child = children_right[parent_node] == curr_node
            assert is_left_child ^ is_right_child

            if is_left_child:
                direction = "left"
            else:
                direction = "right"
            curr_node = parent_node
            curr_feature = features_arr[curr_node]
            curr_threshold = np.round(thresholds_arr[curr_node], 6)
            if important_features_indices is not None:
                curr_feature_original = important_features_indices[curr_feature]
            else:
                curr_feature_original = curr_feature
            path.insert(
                0, (curr_node, curr_feature_original, curr_threshold, direction)
            )
        return path

    def compute_leaf_error_rate_coverage(self, data, labels):
        """Compute error rate and error coverage for every node in the tree."""
        total_failures = np.sum(labels)

        true_dict, false_dict = self.compute_TF_dict(data, labels)

        n_nodes = self.tree_.node_count
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right

        error_rate_array = np.zeros(shape=n_nodes, dtype=float)
        error_coverage_array = np.zeros(shape=n_nodes, dtype=float)

        stack = [(0, True)]
        while len(stack) > 0:
            node_id, traverse = stack.pop()
            child_left = children_left[node_id]
            child_right = children_right[node_id]

            if traverse:
                if child_left != child_right:
                    stack.append((node_id, False))
                    stack.append((child_left, True))
                    stack.append((child_right, True))
                else:
                    num_true_in_node = true_dict[node_id]
                    num_false_in_node = false_dict[node_id]
                    num_total_in_node = num_true_in_node + num_false_in_node

                    if num_total_in_node > 0:
                        leaf_error_rate = num_true_in_node / num_total_in_node
                    else:
                        leaf_error_rate = 0.0
                    leaf_error_coverage = num_true_in_node / total_failures
                    error_coverage_array[node_id] = leaf_error_coverage
                    error_rate_array[node_id] = leaf_error_rate
            else:
                child_left_ER = error_rate_array[child_left]
                child_right_ER = error_rate_array[child_right]

                child_left_EC = error_coverage_array[child_left]
                child_right_EC = error_coverage_array[child_right]

                child_ER = (
                    child_left_ER * child_left_EC + child_right_ER * child_right_EC
                )
                child_EC = child_left_EC + child_right_EC

                if child_EC > 0:
                    error_rate_array[node_id] = child_ER / child_EC
                else:
                    error_rate_array[node_id] = 0.0
                error_coverage_array[node_id] = child_EC

        return error_rate_array, error_coverage_array


def _train_decision_tree(
    train_sparse_features, train_failure, max_depth=1, criterion="entropy"
):
    num_true = np.sum(train_failure)
    num_false = np.sum(np.logical_not(train_failure))
    rel_weight = num_false / num_true
    class_weight_dict = {0: 1, 1: rel_weight}

    decision_tree = BarlowDecisionTreeClassifier(
        max_depth=max_depth, criterion=criterion, class_weight=class_weight_dict
    )
    decision_tree.fit_tree(train_sparse_features, train_failure)
    return decision_tree


def _important_leaf_nodes(decision_tree, precision_array, recall_array):
    """
    Select leaf nodes with highest importance value i.e highest contribution to
    average leaf error rate.
    """
    leaf_ids = decision_tree.leaf_ids
    leaf_precision = precision_array[leaf_ids]
    leaf_recall = recall_array[leaf_ids]
    leaf_precision_recall = leaf_precision * leaf_recall

    important_leaves = np.argsort(-leaf_precision_recall)
    return leaf_ids[important_leaves]
