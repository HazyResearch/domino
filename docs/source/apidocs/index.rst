API Reference
==============


.. toctree::
   :maxdepth: 2
   :hidden:

   embed
   slice
   describe
   train
   gui


The `domino` API Reference is organized into the following sections:

* Slice Discovery Methods
   Most slice discovery methods adhere to a three-step procedure: (1) embed, (2) slice, and (3) describe [eyuboglu_2022]_. For each of these steps, the `domino` package provides implementations of various algorithms under a common API. This makes it easy to compose a custom slice discovery method from different choices for each step.   

   * :ref:`embed-reference` validation data in a representation space. This reference page describes a number of popular encoders implemented in `domino`.

   * :ref:`slice-reference` the representation space into underperforming regions. This reference describes the slicing algorithms implemented in `domino`.   

   * :ref:`describe-reference` slices with natural language. This reference page describes the `domino` method for describing discovered slices.

* Evaluation
   `domino` also provides scripts for performing evaluations of slice discovery methods. These scripts are provided as part of the `domino.eval` sub-package. 

   * :ref:`train-reference`

* Utilities
   * :ref:`gui-reference` - `domino` includes implementations of simple Jupyter Notebook interfaces to help you explore discovered slices.  


.. [eyuboglu_2022]

      Eyuboglu, S. et al. Domino: Discovering Systematic Errors with Cross-Modal Embeddings. in International Conference on Learning Representations (2022).
