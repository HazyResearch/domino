import os
import re
from collections import Counter
from functools import partial
from typing import Mapping, Sequence

import meerkat as mk
import spacy
import terra
import torch
from nltk.tokenize import sent_tokenize
from torch import nn
from torch._C import Value
from torchvision import models, transforms

from domino.multimodal import Classifier
from domino.utils import nested_getattr

nlp = spacy.load("en_core_web_sm")

import pandas as pd
from tqdm import tqdm


@terra.Task
def embed_eeg_text(
    dp: mk.DataPanel,
    model: Classifier,
    layers: Mapping[str, nn.Module],
    reduction_name: Sequence[str] = "mean",
    col_name: str = "input",
    batch_size: int = 128,
    num_workers: int = 4,
    mmap: bool = False,
    device: int = 0,
    run_dir: str = None,
    **kwargs,
) -> mk.DataPanel:

    col = dp[col_name]

    layers = {name: nested_getattr(model, layer) for name, layer in layers.items()}

    class ActivationExtractor:
        """Class for extracting activations a targetted intermediate layer"""

        def __init__(self, reduction_fn: callable = None):
            self.activation = None
            self.reduction_fn = reduction_fn

        def add_hook(self, module, input, output):
            if self.reduction_fn is not None:
                output = self.reduction_fn(output)
            self.activation = output

    layer_to_extractor = {}

    for name, layer in layers.items():
        extractor = ActivationExtractor()
        layer.register_forward_hook(extractor.add_hook)
        layer_to_extractor[name] = extractor

    @torch.no_grad()
    def _score(batch: mk.TensorColumn):
        x = batch.data.to(device)
        model(x)  # Run forward pass

        return {
            name: extractor.activation.cpu().detach().numpy()
            for name, extractor in layer_to_extractor.items()
        }

    model.to(device)
    model.eval()
    emb_dp = col.map(
        _score,
        pbar=True,
        is_batched_fn=True,
        batch_size=batch_size,
        num_workers=num_workers,
        mmap=mmap,
        mmap_path=os.path.join(run_dir, "emb_mmap.npy"),
        flush_size=128,
    )
    return mk.concat([dp, emb_dp], axis=1)


@terra.Task
def embed_words(
    words_dp: mk.DataPanel,
    model: Classifier,
    batch_size: int = 128,
    device: int = 0,
    run_dir: str = None,
) -> mk.DataPanel:
    def encode_word(word):
        return model.text_model(word).squeeze()

    model = model.to(device)
    words_dp["emb"] = words_dp["word"].map(
        lambda x: encode_word(x).cpu().detach().numpy(),
        pbar=True,
        is_batched_fn=False,
        batch_size=batch_size,
    )
    return words_dp


def split_sentence(sentence):
    # return list(filter(lambda x: len(x) > 0, re.split("\W+", sentence.lower())))
    # exp = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    # exp = "\W+"

    # return list(filter(lambda x: len(x) > 0, re.split(exp, sentence.lower())))
    doc = nlp(sentence)

    return list(doc.sents)
    # return sent_tokenize(sentence)


def generate_words_dp(all_reports, min_threshold):
    """
    Return {token: index} for all train tokens (words) that occur min_threshold times or more,
        `index` should be from 0 to N, where N is a number of unique tokens in the resulting dictionary.
    """
    # # convert the list of whole captions to one string
    # concat_str = " ".join([str(elem).strip("\n") for elem in all_reports])
    # # divide the string tokens (individual words), by calling the split_sentence function
    # individual_words = split_sentence(concat_str)
    # # create a list of words that happen min_threshold times or more in that string

    sentences = []
    for report in tqdm(all_reports):
        for x in list(nlp(report).sents):
            sentences.append(str(x))

    words = [key for key, value in Counter(sentences).items() if value >= min_threshold]
    frequency = [
        value for key, value in Counter(sentences).items() if value >= min_threshold
    ]

    # generate the vocabulary(dictionary)
    words_dp = pd.DataFrame({"word": words, "frequency": frequency}).sort_values(
        by="frequency", ascending=False
    )

    return mk.DataPanel.from_pandas(words_dp)
