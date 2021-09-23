import os
from typing import Iterable, List

import clip
import meerkat as mk
import meerkat.nn as mknn
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from terra import Task
from tqdm.auto import tqdm

from domino.utils import batched_pearsonr, merge_in_split


@Task
def embed_words(
    words_dp: mk.DataPanel,
    batch_size: int = 128,
    model: str = "ViT-B/32",
    run_dir: str = None,
) -> mk.DataPanel:
    model, _ = clip.load(model, device=torch.device(0))
    words_dp["tokens"] = mk.LambdaColumn(
        words_dp["word"], fn=lambda x: clip.tokenize(f"a photo of {x}").squeeze()
    )
    words_dp["emb"] = words_dp["tokens"].map(
        lambda x: model.encode_text(x.data.to(0)).cpu().detach().numpy(),
        pbar=True,
        is_batched_fn=True,
        batch_size=batch_size,
    )
    return words_dp


@Task
def embed_images(
    dp: mk.DataPanel,
    img_column: str,
    batch_size: int = 128,
    num_workers: int = 4,
    model: str = "ViT-B/32",
    mmap: bool = False,
    split_dp: mk.DataPanel = None,
    splits: Iterable[str] = None,
    run_dir: str = None,
    **kwargs,
) -> mk.DataPanel:
    if splits is not None:
        dp = merge_in_split(dp, split_dp)
        dp = dp.lz[dp["split"].isin(splits)]

    model, preprocess = clip.load(model, device=torch.device(0))

    dp["__embed_images_input__"] = dp[img_column].to_lambda(preprocess)
    with torch.no_grad():
        dp["emb"] = dp["__embed_images_input__"].map(
            lambda x: model.encode_image(x.data.to(0)).cpu().detach().numpy(),
            pbar=True,
            is_batched_fn=True,
            batch_size=batch_size,
            num_workers=num_workers,
            mmap=mmap,
            mmap_path=os.path.join(run_dir, "emb_mmap.npy"),
            flush_size=128,
        )
    return dp


@Task
def pca_embeddings(
    images_dp: mk.DataPanel,
    words_dp: mk.DataPanel,
    n_components: int = 128,
    top_k: int = 10_000,
):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    pca.fit(words_dp["emb"].lz[:top_k])
    images_dp[f"emb_{n_components}"] = images_dp["emb"].map(
        lambda x: pca.transform(x),
        is_batched_fn=True,
        batch_size=1024,
        mmap=True,
        pbar=True,
    )
    return images_dp


@Task
def get_wiki_words(top_k: int = 1e5, eng_only: bool = False, run_dir: str = None):
    df = pd.read_csv(
        "https://github.com/IlyaSemenov/wikipedia-word-frequency/raw/master/results/enwiki-20190320-words-frequency.txt",
        delimiter=" ",
        names=["word", "frequency"],
    )

    if eng_only:
        import nltk
        from nltk.corpus import words

        nltk.download("words")

        eng_words = words.words()
        eng_df = pd.DataFrame({"word": eng_words})
        df = df.merge(eng_df, how="inner", on="word")

    df = df.sort_values("frequency", ascending=False)
    df = df.drop_duplicates(subset=["word"])
    return mk.DataPanel.from_pandas(df.iloc[: int(top_k)])


def find_explanatory_words(
    target_column: str,
    data_df: pd.DataFrame,
    text_embs: torch.Tensor,
    idx_to_word: dict,
    img_embs: torch.Tensor,
    idx_to_img_id: dict,
    condition_column: str = None,
    condition_value: int = 1,
    top_k: str = 20,
):
    print("computing cosine similarity...")
    img_embs_norm = img_embs / img_embs.norm(dim=-1, keepdim=True)
    text_embs_norm = text_embs / text_embs.norm(dim=-1, keepdim=True)
    similarity = (
        (100.0 * img_embs_norm.to(0) @ text_embs_norm.T.to(0)).softmax(dim=-1).cpu()
    )

    x = similarity.cpu().numpy()
    df = data_df.merge(pd.DataFrame({"file": idx_to_img_id}), on="file", how="right")
    y = torch.tensor(df[target_column]).to(float)

    if condition_column is not None:
        mask = df[condition_column] == condition_value
        x = x[mask]
        y = y[mask]

    corr = batched_pearsonr(torch.tensor(x).to(float), y, batch_first=False)
    dfs = []
    for descending in [True, False]:
        sorted_vals, sorted_idxs = corr.squeeze().sort(descending=descending)
        dfs.append(
            pd.DataFrame(
                {
                    "attribute": target_column,
                    "rank": np.arange(top_k),
                    "descending": descending,
                    "r": sorted_vals[:top_k],
                    "term": [
                        (idx_to_word[idx]).split(" ")[-1] for idx in sorted_idxs[:top_k]
                    ],
                }
            )
        )
    return pd.concat(dfs)


def generate_phrases(word_dp: mk.DataPanel, templates: List[str], device: int = 0):
    from transformers import BertForMaskedLM, BertModel, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    model = BertForMaskedLM.from_pretrained("bert-large-uncased").to(device)

    def _forward(words):
        input_phrases = [
            template.format(word) for word in words for template in templates
        ]
        inputs = tokenizer(input_phrases, return_tensors="pt", padding=True).to(device)
        input_ids = inputs["input_ids"]
        outputs = model(**inputs)
        sorted_preds, sorted_ids = outputs.logits.sort(dim=-1, descending=True)

        probs = []
        output_phrases = []
        for rank in range(10):
            curr_ids = sorted_ids[:, :, rank]
            curr_ids[input_ids != 103] = input_ids[input_ids != 103]

            probs.extend(sorted_preds[:, :, rank].mean(axis=1).cpu().detach().numpy())
            for sent_idx in range(sorted_ids.shape[0]):
                output_phrases.append(
                    tokenizer.decode(
                        sorted_ids[sent_idx, :, rank], skip_special_tokens=True
                    )
                )
        return {"prob": probs, "output_phrase": output_phrases}

    return word_dp["word"].map(_forward, is_batched_fn=True, batch_size=128, pbar=True)
