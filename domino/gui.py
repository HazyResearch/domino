from typing import List, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import meerkat as mk
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from domino.utils import unpack_args
from ._describe import describe


def explore(
    data: mk.DataPanel = None,
    embeddings: Union[str, np.ndarray] = "embedding",
    targets: Union[str, np.ndarray] = "target",
    pred_probs: Union[str, np.ndarray] = "pred_prob",
    slices: Union[str, np.ndarray] = "slices",
    text: mk.DataPanel = None,
    text_embeddings: Union[str, np.ndarray] = "embedding",
    phrase: Union[str, np.ndarray] = "output_phrase",
) -> None:
    """Creates a IPyWidget GUI for exploring discovered slices. The GUI includes two
    sections: (1) The first section displays data visualizations summarizing the
    model predictions and accuracy stratified by slice. (2) The second section displays
    a table (i.e. Meerkat DataPanel) of the data examples most representative of each
    slice. The DataPanel passed to ``data`` should include columns for embeddings,
    targets, pred_probs and slices. Any additional columns will be included in the
    visualization in section (2).

    .. caution::
        This GUI works best in the original Jupyter Notebook, and may not work properly
        in a Jupyter Lab or VSCode environment.

    Args:
        data (mk.DataPanel, optional): A `Meerkat DataPanel` with columns for
            embeddings, targets, and prediction probabilities. The names of the
            columns can be specified with the ``embeddings``, ``targets``, and
            ``pred_probs`` arguments. Defaults to None.
        embeddings (Union[str, np.ndarray], optional): The name of a column in
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
        slices (str, optional): The name of The name of a column in ``data``
            holding discovered slices. If ``data`` is ``None``, then an
            np.ndarray of shape (num_examples, num_slices). Defaults to "slices".
        text (str, optional): A `Meerkat DataPanel` with columns for text phrases and
            their embeddings. The names of the columns can be specified with the 
            ``text_embeddings`` and ``phrase`` arguments. Defaults to None.
        text_embeddings (Union[str, np.ndarray], optional): The name of a colum in
            ``text`` holding embeddings. If ``text`` is ``None``, then an np.ndarray
            of shape (n_phrases, dimension of embedding). Defaults to "embedding".
        phrase (Union[str, np.ndarray], optional): The name of a column in ``text``
            holding text phrases. If ``text`` is ``None``, then an np.ndarray of
            shape (n_phrases,). Defaults to "output_phrase".

    Examples
    --------
     .. code-block:: python
        :name: Example:

        from domino import explore, DominoSDM
        dp = ...  # prepare the dataset as a Meerkat DataPanel

        # split dataset
        valid_dp = dp.lz[dp["split"] == "valid"]
        test_dp = dp.lz[dp["split"] == "test"]

        domino = DominoSDM()
        domino.fit(data=valid_dp)
        test_dp["slices"] = domino.transform(
            data=test_dp, embeddings="emb", targets="target", pred_probs="probs"
        )
        explore(data=test_dp)
    """

    embeddings, targets, pred_probs, slices = unpack_args(
        data, embeddings, targets, pred_probs, slices
    )

    if data is None:
        dp = mk.DataPanel(
            {
                "embeddings": embeddings,
                "targets": targets,
                "pred_probs": pred_probs,
                "domino_slices": slices,
            }
        )
    else:
        dp = data if isinstance(data, mk.DataPanel) else mk.DataPanel(data)

    plot_output = widgets.Output()

    # define functions for generating visualizations
    def plot_slice(slice_idx, slice_threshold: float):

        # TODO (Sabri): Support a confusion matrix for the multiclass case.
        with plot_output:
            plot_df = pd.DataFrame(
                {
                    "in-slice": slices[:, slice_idx].data > slice_threshold,
                    "pred_probs": pred_probs[:, 1].data.numpy()
                    if len(pred_probs.shape) == 2
                    else pred_probs.data,
                    "target": targets.data,
                }
            )
            g = sns.displot(
                data=plot_df,
                hue="in-slice",
                x="pred_probs",
                col="target",
                aspect=1.7,
                height=2,
                facet_kws={"sharey": False},
                hue_order=[False, True],
                palette=["#bdbdbd", "#2396f3"],
                stat="percent",
                common_norm=False,
                bins=20,
            )
            g.set_axis_labels("Model's output probability", "% of examples")
            for target in np.unique(targets.data):
                in_slice = np.sum(
                    (slices[:, slice_idx].data > slice_threshold)
                    & (targets.data == target)
                )
                g.axes[0, int(target)].set_title(
                    f"target={target} \n (# of examples in-slice={in_slice})"
                )

            plot_output.clear_output(wait=True)
            plt.show()

    description_output = widgets.Output()

    def show_descriptions(slice_idx: int, slice_threshold: float):
        description_output.clear_output(wait=False)
        if text is not None:
            description_dp = describe(
                data=dp,
                embeddings=embeddings,
                targets=targets,
                slices=slices,
                slice_idx=slice_idx,
                text=text,
                text_embeddings=text_embeddings,
                phrases=phrase,
                slice_threshold=slice_threshold,
            )
            with description_output:
                display(description_dp[(-description_dp["score"]).argsort()[:5]])

    dp_output = widgets.Output()

    def show_dp(
        slice_idx,
        page_idx: int,
        page_size: int,
        columns: List[str],
        slice_threshold: float,
    ):
        mk.config.DisplayOptions.max_rows = page_size
        dp_output.clear_output(wait=False)
        num_examples_in_slice = np.sum(slices[:, slice_idx].data > slice_threshold)

        with dp_output:
            display(
                dp.lz[
                    (-slices[:, slice_idx]).argsort()[
                        page_size
                        * page_idx : min(
                            page_size * (page_idx + 1), num_examples_in_slice
                        )
                    ]
                ][list(columns)]
            )

    # Create widgets
    slice_idx_widget = widgets.Dropdown(
        value=1,
        options=list(range(slices.shape[-1])),
        description="Slice",
        layout=widgets.Layout(width="150px"),
    )
    slice_threshold_widget = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1.0,
        step=0.025,
        description="Slice Inclusion Threshold",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".3f",
        style={"description_width": "initial"},
    )

    # TODO(Sabri): Add a widget for the # of examples in the slice at the current
    # threshold. It will have to be linked with the threshold widget above.

    column_selector = widgets.SelectMultiple(
        options=dp.columns, value=dp.columns, description="Columns", disabled=False
    )

    page_size_widget = widgets.RadioButtons(
        options=[10, 25, 50], description="Page size"
    )
    page_idx_widget = widgets.BoundedIntText(
        value=0,
        min=0,
        max=10,
        step=1,
        description="Page",
        disabled=False,
        readout=True,
        readout_format="d",
        layout=widgets.Layout(width="150px"),
    )

    # Establish interactions between widgets and the visualization functions
    widgets.interactive(
        show_descriptions,
        slice_idx=slice_idx_widget,
        slice_threshold=slice_threshold_widget,
    )

    widgets.interactive(
        show_dp,
        slice_idx=slice_idx_widget,
        columns=column_selector,
        page_idx=page_idx_widget,
        page_size=page_size_widget,
        slice_threshold=slice_threshold_widget,
    )

    widgets.interactive(
        plot_slice,
        slice_idx=slice_idx_widget,
        slice_threshold=slice_threshold_widget,
    )

    # Layout and display the widgets
    display(
        widgets.HBox(
            [
                widgets.HTML(value="<p><strong> Domino Slice Explorer </strong></p>"),
                slice_idx_widget,
            ]
        )
    )
    display(slice_threshold_widget)
    display(plot_output)
    display(
        widgets.VBox(
            [
                widgets.HTML(
                    value=(
                        "<p> <strong> Natural language descriptions of the slice: "
                        "</strong> </p>"
                    )
                ),
                description_output,
            ]
        )
    )

    display(
        widgets.HBox(
            [
                widgets.VBox(
                    [
                        widgets.HTML(
                            value=(
                                "<style>p{word-wrap: break-word}</style> <p>"
                                + "Select multiple columns with <em>cmd-click</em>."
                                + " </p>"
                            )
                        ),
                        column_selector,
                    ]
                ),
                widgets.VBox([page_idx_widget, page_size_widget]),
            ],
        )
    )
    display(
        widgets.VBox(
            [
                widgets.HTML(
                    value=(
                        "<p> <strong>  Examples in the slice, ranked by likelihood: "
                        "</strong> </p>"
                    )
                ),
                dp_output,
            ]
        )
    )

    # To actually run the functions `plot_slice` and `show_dp` we need update the value
    # of one of the widgets.
    slice_idx_widget.value = 0
