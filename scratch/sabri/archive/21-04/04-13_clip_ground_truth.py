import os

import pandas as pd
from terra import Task

import domino.clip

os.environ["TERRA_CONFIG_PATH"] = "/home/sabri/code/domino-21/terra_config.json"


@Task.make_task
def evaluate_clip_on_ground_truth_slices(
    text_embs,
    idx_to_word,
    img_embs,
    idx_to_img_id,
    run_dir: str = None,
):
    import numpy as np
    import torch
    from tqdm.auto import tqdm

    from domino.data.celeb import ATTRIBUTES, build_celeb_df

    print("computing cosine similarity...")
    img_embs_norm = img_embs / img_embs.norm(dim=-1, keepdim=True)
    text_embs_norm = text_embs / text_embs.norm(dim=-1, keepdim=True)
    similarity = (
        (100.0 * img_embs_norm.to(0) @ text_embs_norm.T.to(0)).softmax(dim=-1).cpu()
    )
    x = similarity.cpu().numpy()

    print("getting top terms for each attribute...")
    data_df = build_celeb_df.out(141).load()
    top_k = 10
    dfs = []
    for attribute in tqdm(ATTRIBUTES):
        y = torch.tensor(
            data_df.merge(
                pd.DataFrame({"file": idx_to_img_id}), on="file", how="right"
            )[attribute]
        ).to(float)
        corr = pearsonr(torch.tensor(x).to(float), y, batch_first=False)
        for descending in [True, False]:
            sorted_vals, sorted_idxs = corr.squeeze().sort(descending=descending)
            dfs.append(
                pd.DataFrame(
                    {
                        "attribute": attribute,
                        "rank": np.arange(top_k),
                        "descending": descending,
                        "r": sorted_vals[:top_k],
                        "term": [
                            (idx_to_word[idx]).split(" ")[-1]
                            for idx in sorted_idxs[:top_k]
                        ],
                    }
                )
            )

    df = pd.concat(dfs)
    return df


if __name__ == "__main__":
    text_embs, idx_to_word = domino.clip.embed_words.out(189)
    img_embs, idx_to_img_id = domino.clip.embed_images.out(178)
    evaluate_clip_on_ground_truth_slices(
        text_embs=text_embs,
        idx_to_word=idx_to_word,
        img_embs=img_embs,
        idx_to_img_id=idx_to_img_id,
    )
