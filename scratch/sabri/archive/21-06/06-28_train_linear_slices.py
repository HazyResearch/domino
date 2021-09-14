import numpy as np
import pandas as pd
import terra

from domino.evaluate.train import train_linear_slices

train_linear_slices(
    dp_run_id=542,
    target_correlate_pairs=[
        ("male", "smiling"),
        ("arched_eyebrows", "black_hair"),
        ("eyeglasses", "wearing_necktie"),
        ("wearing_necktie", "eyeglasses"),
        ("mustache", "wearing_hat"),
        ("sideburns", "mustache"),
        ("young", "no_beard"),
        ("receding_hairline", "wearing_necktie"),
        ("bushy_eyebrows", "5_o_clock_shadow"),
        ("wavy_hair", "male"),
        ("wearing_necklace", "blond_hair"),
    ],
    batch_size=196,
    num_workers=6,
    max_epochs=5,
    val_check_interval=50,
    num_samples=1,
)
