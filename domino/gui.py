from typing import List, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import meerkat as mk
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display


def explore(
    data: Union[dict, mk.DataPanel] = None,
    embeddings: Union[str, np.ndarray] = "embedding",
    targets: Union[str, np.ndarray] = "target",
    pred_probs: Union[str, np.ndarray] = "pred_probs",
    slices: Union[str, np.ndarray] = "slices",
) -> None:
    """Creates a IPyWidget GUI for exploring discovered slices. The GUI includes two
    sections: (1) The first section displays data visualizations summarizing the predictions
    model predictions and accuracy stratified by slice. (2) The second section displays
    a table (i.e. Meerkat DataPanel) of the data examples most representative of each
    slice. The DataPane passed to ``data`` should include columns for embeddings,
    targets, pred_probs and slices. Any additional columns will be included in the
    visualization in section (2).

    Example::
        from domino import gui, DominoSDM
        dp = ...  # prepare the dataset as a Meerkat DataPanel

        domino = DominoSDM()
        domino.fit(data=dp)
        dp["slices"] = domino.transform(
            data=dp, embeddings="emb", targets="target", pred_probs="probs"
        )

    .. note::
        This function works best in the original Jupyter Notebook, and has not been
        tested thoroughly in a Jupyter Lab or VSCode environment.

    Args:
        data (Union[dict, mk.DataPanel], optional): A Meerkat DataPanel holding the
            dataset. Should include columns for embeddings, targets, pred_probs and
            slices, as described below. Any additional columns will be included in the
            visualization in section (2). Defaults to None.
        embeddings (Union[str, np.ndarray], optional): The name of the embedding column
            in ``data`` or, if ``data`` is ``None``, then embeddings as an np.ndarray
            of shape (num_examples, embedding_dimension). Defaults to "embedding".
        targets (Union[str, np.ndarray], optional): The name of the target column in
            ``data`` or, if ``data`` is ``None``, then the targets as an np.ndarray of
            shape (num_examples,). Defaults to "target".
        pred_probs (Union[str, np.ndarray], optional): The name of the
            predicted probability column in ``data`` or, if ``data`` is ``None``, then
            the predicted probabilities as an np.ndarray of shape (num_examples,
            num_classes). Defaults to "pred_probs".
        slices (str, optional): The name of the column in ``data`` holding the
            discovered slices or, if ``data`` is ``None``, then the slices as an
            np.ndarray of shape (num_examples, num_slices). Defaults to "slices".

    """
    if data is None and any(
        map(lambda x: isinstance(x, str), [embeddings, targets, pred_probs, slices])
    ):
        raise ValueError(
            "If `embeddings`, `target` or `pred_probs` are strings, `data`"
            " must be provided."
        )

    embeddings = data[embeddings] if isinstance(embeddings, str) else embeddings
    targets = data[targets] if isinstance(targets, str) else targets
    pred_probs = data[pred_probs] if isinstance(pred_probs, str) else pred_probs
    slices = data[slices] if isinstance(slices, str) else slices

    if data is None:
        dp = mk.DataPanel(
            {
                "embeddings": embeddings,
                "targets": targets,
                "pred_probs": pred_probs,
                "domino_slices)": slices,
            }
        )
    else:
        dp = data if isinstance(data, mk.DataPanel) else mk.DataPanel(data)

    plot_output = widgets.Output()

    # define functions for generating visualizations
    def plot_slice(slice_idx, slice_threshold: float):

        # TODO (Sabri): Support a confusion matrix for the multiclass case.
        with plot_output:
            data = pd.DataFrame(
                {
                    "in-slice": slices[:, slice_idx].data > slice_threshold,
                    "pred_probs": pred_probs[:, 1].data.numpy(),
                    "target": targets.data,
                }
            )
            g = sns.displot(
                data=data,
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
            )
            g.set_axis_labels("Model's output probability", "% of examples")

            plot_output.clear_output(wait=True)
            plt.show()

    dp_output = widgets.Output()

    def show_dp(slice_idx, columns: List[str], page_idx: int, page_size: int):
        mk.config.DisplayOptions.max_rows = page_size
        dp_output.clear_output(wait=False)

        with dp_output:
            display(
                dp.lz[
                    (-dp["domino_slices"][:, slice_idx]).argsort()[
                        page_size * page_idx : page_size * (page_idx + 1)
                    ]
                ][list(columns)]
            )

    # Create widgets
    slice_idx_widget = widgets.Dropdown(
        value=1,
        options=list(range(dp["domino_slices"].shape[-1])),
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
        show_dp,
        slice_idx=slice_idx_widget,
        columns=column_selector,
        page_idx=page_idx_widget,
        page_size=page_size_widget,
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
    display(dp_output)

    # To actually run the functions `plot_slice` and `show_dp` we need update the value
    # of one of the widgets.
    slice_idx_widget.value = 0
