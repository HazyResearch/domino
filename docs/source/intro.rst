💡 What is dcbench?
-------------------

This benchmark evaluates the steps in your machine learning workflow beyond model training and tuning. This includes feature cleaning, slice discovery, and coreset selection. We call these “data-centric” tasks because they're focused on exploring and manipulating data – not training models. ``dcbench`` supports a growing number of them:

* :any:`minidata`: Find the smallest subset of training data on which a fixed model architecture achieves accuracy above a threshold. 
* :any:`slice_discovery`: Identify subgroups on which a model underperforms.
* :any:`budgetclean`: Given a fixed budget, clean input features of training data to improve model performance.  


``dcbench`` includes tasks that look very different from one another: the inputs and
outputs of the slice discovery task are not the same as those of the
minimal data cleaning task. However, we think it important that
researchers and practitioners be able to run evaluations on data-centric
tasks across the ML lifecycle without having to learn a bunch of
different APIs or rewrite evaluation scripts.

So, ``dcbench`` is designed to be a common home for these diverse, but
related, tasks. In ``dcbench`` all of these tasks are structured in a
similar manner and they are supported by a common Python API that makes
it easy to download data, run evaluations, and compare methods.

