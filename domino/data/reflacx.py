import os
import terra
from meerkat import DataPanel
import pandas as pd
import numpy as np
import pickle
import shutil


def build_reflacx_dp(data_dir):

    metadata_pth = os.path.join(data_dir, "main_data/metadata_phase_3.csv")
    metadata_df = pd.read_csv(metadata_pth)

    # remove rows without gaze data
    gaze_mask = metadata_df["eye_tracking_data_discarded"] == False
    metadata_df = metadata_df[gaze_mask]

    # retrieve gaze data and transcriptions
    reflacx_data = []
    for index, row in metadata_df.iterrows():
        entry_id = row["id"]
        annot_dir = os.path.join(data_dir, f"main_data/{entry_id}")
        fixation_df = pd.read_csv(os.path.join(annot_dir, "fixations.csv"))
        # TODO: scale x_pos and y_pos appropriately
        time = (
            fixation_df["timestamp_end_fixation"]
            - fixation_df["timestamp_start_fixation"]
        )
        x_pos = fixation_df["x_position"]
        y_pos = fixation_df["y_position"]
        pupil = fixation_df["pupil_area_normalized"]
        gaze_seq = np.array([time, x_pos, y_pos, pupil]).T
        row["gaze_seq"] = gaze_seq

        with open(os.path.join(annot_dir, "transcription.txt")) as f:
            lines = f.readlines()

        row["transcription"] = lines[0]
        reflacx_data.append(dict(row))

    dp = DataPanel(reflacx_data)

    return dp


def copy_images_from_mimic(mimic_pth, save_pth, img_list_pth):

    with open(img_list_pth) as f:
        img_pths = f.readlines()

    for pth in img_pths:
        pth = pth.strip("\n")
        mimic_pth_ = os.path.join(mimic_pth, pth.split("2.0.0/")[-1])
        shutil.copyfile(mimic_pth_, save_pth)
        breakpoint()

    return None


def main():
    data_dir = "/data/ssd1crypt/datasets/reflacx"
    dp = build_reflacx_dp(data_dir)

    # save image paths
    img_pths = dp["image"].data
    txtfile = open("reflacx_img_pths.txt", "w")
    for pth in img_pths:
        txtfile.write(pth + "\n")
    txtfile.close()


if __name__ == "__main__":
    # main()
    copy_images_from_mimic(
        "/dfs/scratch1/common/public-datasets/mimic-cxr-2.0.0.physionet.org",
        "/dfs/scratch1/ksaab/data/reflacx_images",
        "/home/ksaab/Documents/domino/domino/data/reflacx_img_pths.txt",
    )

