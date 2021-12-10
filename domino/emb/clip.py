import os
import meerkat as mk
import torch
import clip


def embed_images(
    data: mk.DataPanel,
    image: str = "image",
    batch_size: int = 128,
    num_workers: int = 4,
    model: str = "ViT-B/32",
    mmap_dir: str = None,
    **kwargs,
) -> mk.DataPanel:

    model, preprocess = clip.load(model, device=torch.device(0))

    data["__embed_images_input__"] = data[image].to_lambda(preprocess)
    with torch.no_grad():
        data["emb"] = data["__embed_images_input__"].map(
            lambda x: model.encode_image(x.data.to(0)).cpu().detach().numpy(),
            pbar=True,
            is_batched_fn=True,
            batch_size=batch_size,
            num_workers=num_workers,
            mmap=mmap_dir is None,
            mmap_path=None
            if mmap_dir is None
            else os.path.join(mmap_dir, "emb_mmap.npy"),
            flush_size=128,
        )
    return data
