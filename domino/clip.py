import pandas as pd
from terra import Task
import numpy as np
import torch
import pandas as pd
import clip
from tqdm.auto import tqdm
import robustnessgym as rg
import torchvision.transforms as transforms
from domino.utils import batched_pearsonr


@Task.make_task
def embed_words(
    df: pd.DataFrame,
    batch_size: int = 128,
    model: str = "/afs/cs.stanford.edu/u/sabrieyuboglu/data/models/clip/RN50.pt",
    top_k: int = None,
    run_dir: str = None,
):
    if top_k is not None:
        df = df.sort_values("frequency", ascending=False)
        df = df.iloc[:top_k]
    model, _ = clip.load(model, device=torch.device(0), jit=False)
    embs = []
    texts = []
    with torch.no_grad():
        for _, batch_df in tqdm(df.groupby(np.arange(len(df)) // batch_size)):
            text_samples = list(map(lambda x: f"a photo of {x}", batch_df.word))
            tokens = clip.tokenize(text_samples).to(0)
            emb = model.encode_text(tokens)
            embs.append(emb.cpu())
            texts.extend(text_samples)
    return torch.cat(embs, dim=0), texts


@Task.make_task
def embed_images(
    data_df: pd.DataFrame,
    img_column: str,
    id_column: str,
    img_transform: callable,
    split: str = "valid",
    batch_size: int = 128,
    num_workers: int = 4,
    model: str = "/afs/cs.stanford.edu/u/sabrieyuboglu/data/models/clip/RN50.pt",
    run_dir: str = None,
    **kwargs,
):

    model, preprocess = clip.load(model, device=torch.device(0), jit=False)
    transform = transforms.Compose(
        [
            img_transform,
            transforms.Lambda(lambda x: x.to(torch.uint8)),
            transforms.ToPILImage(),
            preprocess,
        ]
    )
    dataset = rg.Dataset.load_image_dataset(
        data_df[data_df.split == split].to_dict("records"),
        img_columns=img_column,
        transform=transform,
    )
    dl = dataset.to_dataloader(
        columns=[img_column, id_column],
        num_workers=num_workers,
        batch_size=batch_size,
    )
    embs = []
    img_ids = []
    with torch.no_grad():
        for img, img_id in tqdm(dl):
            img = img.to(0)
            emb = model.encode_image(img)
            embs.append(emb.cpu())
            img_ids.extend(img_id)
    return torch.cat(embs, dim=0), img_ids


@Task.make_task
def get_wiki_words(top_k: int = 1e5, eng_only: bool = False, run_dir: str = None):
    df = pd.read_csv(
        "https://github.com/IlyaSemenov/wikipedia-word-frequency/raw/master/results/enwiki-20190320-words-frequency.txt",
        delimiter=" ",
        names=["word", "frequency"],
    )

    if eng_only:
        from nltk.corpus import words

        eng_words = words.words()
        eng_df = pd.DataFrame({"word": eng_words})
        df = df.merge(eng_df, how="inner", on="word")

    df = df.sort_values("frequency", ascending=False)
    return df.iloc[: int(top_k)]


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
