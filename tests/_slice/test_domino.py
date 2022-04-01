from sklearn import metrics

from domino import DominoSlicer

from ..testbeds import SliceTestBed


def test_domino_results():

    testbed = SliceTestBed(length=9)

    domino = DominoSlicer(
        n_slices=3,
        n_mixture_components=3,
        n_pca_components=None,
        init_params="random",
        random_state=42,
    )
    domino.fit(data=testbed.dp)

    pred_slices = domino.predict(data=testbed.dp)
    assert metrics.rand_score(testbed.clusters, pred_slices.argmax(axis=-1)) == 1.0

    prob_slices = domino.predict_proba(data=testbed.dp)
    assert (pred_slices.argmax(axis=-1) == prob_slices.argmax(axis=-1)).all()
