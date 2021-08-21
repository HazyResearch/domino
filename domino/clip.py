import clip
import meerkat as mk
import meerkat.nn as mknn
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from terra import Task
from tqdm.auto import tqdm

from domino.utils import batched_pearsonr


@Task.make_task
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
        lambda x: model.encode_text(x.data.to(0)).cpu().detach(),
        pbar=True,
        is_batched_fn=True,
        batch_size=batch_size,
    )
    return words_dp


def embed_images(
    dp: mk.DataPanel,
    img_column: str,
    batch_size: int = 128,
    num_workers: int = 4,
    model: str = "ViT-B/32",
    run_dir: str = None,
    **kwargs,
):
    model, preprocess = clip.load(model, device=torch.device(0))

    dp["__embed_images_input__"] = dp[img_column].to_lambda(preprocess)
    with torch.no_grad():
        dp["emb"] = dp["__embed_images_input__"].map(
            lambda x: model.encode_image(x.data.to(0)).cpu().detach(),
            pbar=True,
            is_batched_fn=True,
            batch_size=batch_size,
            num_workers=num_workers,
            output_type=mknn.EmbeddingColumn,
        )
    return dp


@Task.make_task
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
