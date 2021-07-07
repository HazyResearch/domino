import numpy as np
import pandas as pd
import terra

from domino.evaluate.train import train_linear_slices

train_linear_slices(
    dp_run_id=1267,
    target_correlate_pairs=[
        ("Lung_Lesion_uzeros", "gender_male"),
        ("Pleural_Effusion_uzeros", "gender_male"),
        ("Edema_uzeros", "gender_male"),
        ("Enlarged_Cardiomediastinum_uzeros", "gender_male"),
        ("Support_Devices_uzeros", "gender_male"),
        ("Edema_uzeros", "ethnicity_black"),
        ("Pleural_Effusion_uzeros", "ethnicity_asian"),
        ("Edema_uzeros", "ethnicity_asian"),
        ("Pleural_Effusion_uzeros", "burned_in_annotation"),
        ("Pleural_Effusion_uzeros", "Support_Devices_uzeros"),
        ("Lung_Lesion_uzeros", "patient_orientation_rf"),
        ("Support_Devices_uzeros", "young"),
    ],
    batch_size=24,
    num_workers=7,
    max_epochs=3,
    input_column="input_224",
    id_column="dicom_id",
    valid_split="validate",
    val_check_interval=20,
    weighted_sampling=True,
    samples_per_epoch=50000,
    config={"lr": 1e-4, "model_name": "resnet", "arch": "resnet50"},
    num_samples=1,
)
