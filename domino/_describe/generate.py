import torch
import meerkat as mk
from itertools import product
import numpy as np
from typing import List
import pandas as pd


PAD_TOKEN_ID = 103


def generate_candidate_descriptions(
    templates: List[str],
    device: int = 0,
    k: int = 2,
    bert_size: str = "base",
    num_candidates: str = 30_000,
    num_seed_words: int = 10_000,
    score_with_gpt: bool = False,
) -> mk.DataPanel:
    words_dp = _get_wiki_words(top_k=num_seed_words, eng_only=True)
    from transformers import BertForMaskedLM, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(f"bert-{bert_size}-uncased")
    model = (
        BertForMaskedLM.from_pretrained(f"bert-{bert_size}-uncased").to(device).eval()
    )

    @torch.no_grad()
    def _forward_mlm(words):
        input_phrases = [
            template.format(word) for word in words for template in templates
        ]
        inputs = tokenizer(input_phrases, return_tensors="pt", padding=True).to(device)
        input_ids = inputs["input_ids"]

        outputs = model(**inputs)  # shape=(num_sents, num_tokens_in_sent, size_vocab)
        probs = torch.softmax(outputs.logits, dim=-1).detach()
        top_k_out = probs.topk(k=k, dim=-1)

        output_phrases = []
        output_probs = []
        for sent_idx in range(probs.shape[0]):
            mask_mask = input_ids[sent_idx] == PAD_TOKEN_ID
            mask_range = torch.arange(mask_mask.sum())
            token_ids = top_k_out.indices[sent_idx, mask_mask]
            token_probs = top_k_out.values[sent_idx, mask_mask]
            for local_idxs in product(np.arange(k), repeat=mask_mask.sum()):
                output_ids = torch.clone(input_ids[sent_idx])
                output_ids[mask_mask] = token_ids[mask_range, local_idxs]
                output_phrases.append(
                    tokenizer.decode(output_ids, skip_special_tokens=True)
                )
                output_probs.append(
                    token_probs[mask_range, local_idxs].mean().cpu().numpy()
                )

        return {"prob": output_probs, "output_phrase": output_phrases}

    candidate_phrases = words_dp["word"].map(
        _forward_mlm, is_batched_fn=True, batch_size=16, pbar=True
    )

    candidate_phrases = (
        candidate_phrases.to_pandas()
        .dropna()
        .sort_values("prob", ascending=False)[:num_candidates]
    )

    if score_with_gpt:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        @torch.no_grad()
        def _forward_lm(phrase):
            tokens_tensor = gpt_tokenizer.encode(
                phrase, add_special_tokens=False, return_tensors="pt"
            ).to(device)
            loss = gpt_model(tokens_tensor, labels=tokens_tensor)[0]
            return {
                "loss": np.exp(loss.cpu().detach().numpy()),
                "output_phrase": phrase,
            }

        # unclear how to get loss for a batch of sentences
        return mk.DataPanel.from_pandas(candidate_phrases)["output_phrase"].map(
            _forward_lm, is_batched_fn=False, pbar=True
        )

    return mk.DataPanel.from_pandas(candidate_phrases)


def _get_wiki_words(top_k: int = 1e5, eng_only: bool = False):
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
