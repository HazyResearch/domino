import numpy as np
from skimage.util.shape import view_as_windows


def make_heatmaps(gaze_seqs, num_patches=8, normalize_heatmaps=False):
    all_grids = np.zeros((len(gaze_seqs), num_patches, num_patches), dtype=np.float32)
    for ndx, gaze_seq in enumerate(gaze_seqs):
        # loop through gaze seq and increment # of visits to each patch
        for (x, y, t) in gaze_seq:
            # make sure if x or y are > 1 then they are 1
            x, y = np.clip([x, y], 0.0, 0.999)
            patch_x, patch_y = int(x * num_patches), int(y * num_patches)
            all_grids[ndx, patch_x, patch_y] += t
        if normalize_heatmaps:
            # Destroy total time information, as a diagnostic
            all_grids[ndx] /= np.sum(all_grids[ndx])
    return all_grids


def max_visit(heatmap, pct=0.5):
    if np.any(heatmap > np.sum(heatmap) * pct):
        return np.max(heatmap)
    return 0


def diffusivity(heatmap, s1=5, s2=5, stride=1):
    heatmap = heatmap / np.sum(heatmap)
    heatmap_windows = view_as_windows(heatmap, (s1, s2), step=stride)
    conv_results = np.tensordot(
        heatmap_windows, np.ones((s1, s2)), axes=((2, 3), (0, 1))
    )
    return np.amax(conv_results)


def unique_visits(heatmap):
    return np.sum(heatmap > 0)


def total_time(heatmap):
    return np.sum(heatmap)


def apply_lf(data, lf):
    # Apply a labeling function to a bunch of data
    return np.array([lf(x) for x in data])
