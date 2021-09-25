import meerkat as mk
import nltk
import terra
from meerkat.contrib.imagenet import build_imagenet_dps
from nltk.corpus import wordnet as wn

from domino.data.gqa import DATASET_DIR

DATASET_DIR = "/home/common/datasets/imagenet"


@terra.Task
def get_imagenet_dp(dataset_dir: str = DATASET_DIR):

    dp = build_imagenet_dps(dataset_dir=dataset_dir, download=False)
    nltk.download("wordnet")

    pos_offset_to_synset = {
        pos_offset: wn.synset_from_pos_and_offset(
            pos=pos_offset[0], offset=int(pos_offset[1:])
        ).name()
        for pos_offset in dp["synset"].unique()
    }

    dp["synset_pos_offset"] = dp["synset"]
    dp["synset"] = dp["synset"].apply(lambda x: pos_offset_to_synset[x])
    return dp
