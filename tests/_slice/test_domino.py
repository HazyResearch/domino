from sklearn import metrics
import pytest
from itertools import product

from domino import DominoSlicer

from ..testbeds import SliceTestBed


@pytest.mark.parametrize(
    "init_params,type", product(["random", "confusion"], ["numpy", "torch"])
)
def test_domino_results(init_params: str, type: str):

    testbed = SliceTestBed(length=9, type=type)

    domino = DominoSlicer(
        n_slices=5,
        n_mixture_components=5,
        n_pca_components=None,
        init_params=init_params,
        random_state=42,
    )
    domino.fit(data=testbed.dp)

    pred_slices = domino.predict(data=testbed.dp)
    assert metrics.rand_score(testbed.clusters, pred_slices.argmax(axis=-1)) == 1.0

    prob_slices = domino.predict_proba(data=testbed.dp)
    assert (pred_slices.argmax(axis=-1) == prob_slices.argmax(axis=-1)).all()
