.. _slice-reference:

Slice
========
This page includes an API reference for the *slicers* provided by ``domino``. Recall that most slice discovey methods adhere to a three-step procedure: (1) embed, (2) **slice**, and (3) describe. In this second step, we search an embedding space for regions where a model underperforms. Algorithms that can perform this step are called *slicers* and, in ``domino``, are subclasses of the abstract :class:`~domino.Slicer`. For example, the :class:`~domino.SpotlightSlicer`, directly optimizes the parameters of Gaussian kernel to highlight regions with a high concentration of errors [deon_2022]_.

All slicers in Domino share a common, sklearn-esque API. They each implement three methods: :meth:`~domino.Slicer.fit`, :meth:`~domino.Slicer.predict`, and :meth:`~domino.Slicer.predict_proba`. 

* :meth:`~domino.Slicer.fit` Learn a set of *slicing functions* that partition the embedding space into regions with high error rates.
* :meth:`~domino.Slicer.predict` Apply the learned slicing functions to data, assigning each datapoint to zero or more slices. 
* :meth:`~domino.Slicer.predict_proba` Apply the learned slicing functions to data, producing "soft" slice assignemnts.

All three methods accept ``embeddings``, ``targets``, ``pred_probs``, and ``losses``. There are two ways to use these arguments:

1. By passing `NumPy arrays <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_ directly. 
2. By passing a `Meerkat DataPanel <https://meerkat.readthedocs.io/en/latest/guide/data_structures.html#datapanel>`_ to the ``data`` argument and string column names to ``embeddings``, ``targets``, ``pred_probs``, and ``losses``. 

Note that not all slicers require all arguments. For example, the :class:`~domino.DominoSlicer` requires the `embeddings`, `target`, and `pred_probs` arguments for :meth:`~domino.Slicer.fit`, but only ``embeddings`` is required for :meth:`~domino.Slicer.predict` and :meth:`~domino.Slicer.predict_proba`.

Consider this simple example where we :meth:`~domino.Slicer.fit` the :class:`~domino.DominoSlicer` on the validation set and apply :meth:`~domino.Slicer.predict` to the test set.

.. code-block:: python

        from domino import DominoSlicer
        dp = ...  # load a dataset with columns "emb", "target" and "pred_probs" into a Meerkat DataPanel

        # split dataset
        valid_dp = dp.lz[dp["split"] == "valid"]
        test_dp = dp.lz[dp["split"] == "test"]

        domino = DominoSlicer()
        domino.fit(
            data=valid_dp, embeddings="emb", targets="target", pred_probs="pred_probs"
        )
        dp["domino_slices"] = domino.predict(
            data=test_dp, embeddings="emb",
        )

Slicers can be configured by passing parameters to the constructor. Each slicer has a different set of parameters; for example, the :class:`~domino.DominoSlicer` has a parameter called ``max_iter`` which controls the maximum number of EM iterations. See the documentation below for the parameters of each slicer. 

To access these parameters from a slicer, users can use :meth:`~domino.Slicer.get_params`, which returns a dictionary mapping parameter names (as defined in the constructor) to values.

.. contents:: Table of Contents
    :depth: 3
    :local:

Abstract Base Class: Slicer
----------------------------
.. autoclass:: domino.Slicer 
    :members:

DominoSlicer
-------------
.. autoclass:: domino.DominoSlicer

SpotlightSlicer
----------------
.. autoclass:: domino.SpotlightSlicer 


MultiaccuracySlicer
--------------------
.. autoclass:: domino.MultiaccuracySlicer

BarlowSlicer
------------
.. autoclass:: domino.BarlowSlicer
