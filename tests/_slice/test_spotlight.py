from sklearn import metrics
import pytest
import numpy as np

from domino import SpotlightSlicer


from ..testbeds import SliceTestBed


@pytest.mark.parametrize("pass_losses", [True, False])
def test_domino_results(pass_losses):

    testbed = SliceTestBed(length=9)

    method = SpotlightSlicer(n_slices=2, n_steps=3)

    if pass_losses:
        kwargs = {"losses": "losses"}
    else:
        kwargs = {"targets": "target", "pred_probs": "pred_probs"}
    method.fit(data=testbed.dp, **kwargs)

    pred_slices = method.predict(data=testbed.dp, **kwargs)
    # assert output is a numpy array
    assert isinstance(pred_slices, np.ndarray)
    # assert that the shape of the array is (n_samples, n_slices)
    assert pred_slices.shape == (len(testbed.dp), 2)

    prob_slices = method.predict_proba(data=testbed.dp, **kwargs)
    # assert output is a numpy array
    assert isinstance(prob_slices, np.ndarray)
    # assert that the shape of the array is (n_samples, n_slices)
    assert prob_slices.shape == (len(testbed.dp), 2)
