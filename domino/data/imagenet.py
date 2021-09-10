import meerkat as mk
import terra
from meerkat.contrib.imagenet import build_imagenet_dps

from domino.data.gqa import DATASET_DIR

DATASET_DIR = "/home/common/datasets/imagenet"


@terra.Task
def get_imagenet_dp(dataset_dir: str = DATASET_DIR):
    return build_imagenet_dps(dataset_dir=dataset_dir, download=False)
