import meerkat as mk
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dosma import DicomReader
from PIL import Image
from tqdm import tqdm


def load_contrastive_dp(dp, num_a=10, num_p=10, num_n=10):
    """
    given a datapanel, create a new dp with
    (anchor,positive,negative) pairs for CnC
    using information from "group_id" column
    """

    # TODO: this code is only for 1 subgroup attribute!!

    # HACK: we are going to assume that:
    #       0: no pmx, no tube  --> majority group
    #       1: pmx, no tube     --> minority group
    #       2: no pmx, tube     --> minority group
    #       3: pmx, tube        --> majority group

    # loop through each anchor, which are only points from groups 0 and 3
    data = []

    positive_entries_0 = dp[dp["group_id"].data == 2]
    negative_entries_0 = dp[dp["group_id"].data == 1]
    positive_entries_3 = dp[dp["group_id"].data == 1]
    negative_entries_3 = dp[dp["group_id"].data == 2]

    # positive_entries_0 = np.random.choice(positive_entries_0, num_p)
    # negative_entries_0 = np.random.choice(negative_entries_0, num_n)
    # positive_entries_3 = np.random.choice(positive_entries_3, num_p)
    # negative_entries_3 = np.random.choice(negative_entries_3, num_p)

    if num_a == -1:
        anchor_idxs = np.arange(len(dp))
    else:
        anchor_idxs = np.random.choice(len(dp), num_a)

    print("-- Creating Contrastive DP --")
    for a_idx in tqdm(anchor_idxs):
        a_idx = int(a_idx)
        if dp[a_idx]["group_id"] == 0:
            # positives are group_id 2 and negatives are group_id 1
            if num_p == -1:
                positive_entries = positive_entries_0
            else:
                positive_entries = np.random.choice(positive_entries_0, num_p)
            if num_n == -1:
                negative_entries = negative_entries_0
            else:
                negative_entries = np.random.choice(negative_entries_0, num_n)
        elif dp[a_idx]["group_id"] == 3:
            # positives are group_id 1 and negatives are group_id 2
            if num_p == -1:
                positive_entries = positive_entries_3
            else:
                positive_entries = np.random.choice(positive_entries_3, num_p)
            if num_n == -1:
                negative_entries = negative_entries_3
            else:
                negative_entries = np.random.choice(negative_entries_3, num_n)

        else:
            continue

        for p_idx in range(len(positive_entries)):
            for n_idx in range(len(negative_entries)):
                contrastive_sample = {
                    "a_filepath": dp[a_idx]["filepath"],
                    "p_filepath": positive_entries[p_idx]["filepath"],
                    "n_filepath": negative_entries[n_idx]["filepath"],
                    "a_target": dp[a_idx]["target"],
                    "p_target": positive_entries[p_idx]["target"],
                    "n_target": negative_entries[n_idx]["target"],
                    "a_group_id": dp[a_idx]["group_id"],
                    "p_group_id": positive_entries[p_idx]["group_id"],
                    "n_group_id": negative_entries[n_idx]["group_id"],
                    "split": dp[a_idx]["split"],
                }
                data.append(contrastive_sample)

    contrastive_dp = mk.DataPanel.from_batch(data)

    input_col_a = contrastive_dp[["a_filepath", "split"]].to_lambda(fn=cxr_loader)
    contrastive_dp.add_column(
        "a_input",
        input_col_a,
        overwrite=True,
    )

    input_col_p = contrastive_dp[["p_filepath", "split"]].to_lambda(fn=cxr_loader)
    contrastive_dp.add_column(
        "p_input",
        input_col_p,
        overwrite=True,
    )

    input_col_n = contrastive_dp[["n_filepath", "split"]].to_lambda(fn=cxr_loader)
    contrastive_dp.add_column(
        "n_input",
        input_col_n,
        overwrite=True,
    )

    return contrastive_dp


CXR_MEAN = 0.48865
CXR_STD = 0.24621
CXR_SIZE = 256
CROP_SIZE = 224


def cxr_pil_loader(input_dict):
    input_keys = list(input_dict.keys())
    filepath_key = input_keys["filepath" in input_keys]
    filepath = input_dict[filepath_key]
    loader = DicomReader(group_by=None, default_ornt=("SI", "AP"))
    volume = loader(filepath)[0]
    array = volume._volume.squeeze()
    return Image.fromarray(np.uint8(array))


def cxr_loader(input_dict):
    train = input_dict["split"] == "train"
    # loader = DicomReader(group_by=None, default_ornt=("SI", "AP"))
    # volume = loader(filepath)
    img = cxr_pil_loader(input_dict)
    if train:
        img = transforms.Compose(
            [
                transforms.Resize(CXR_SIZE),
                transforms.RandomCrop(CROP_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CXR_MEAN, CXR_STD),
            ]
        )(img)
    else:
        img = transforms.Compose(
            [
                transforms.Resize([CROP_SIZE, CROP_SIZE]),
                transforms.ToTensor(),
                transforms.Normalize(CXR_MEAN, CXR_STD),
            ]
        )(img)
    return img.repeat([3, 1, 1])


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, config):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = config["temperature"]

        self.sim = nn.CosineSimilarity(dim=1)

    def forward(self, encoder, contrastive_batch):

        a_inputs, p_inputs, n_inputs = contrastive_batch
        a_outputs, p_outputs, n_outputs = (
            encoder(a_inputs).squeeze(),
            encoder(p_inputs).squeeze(),
            encoder(n_inputs).squeeze(),
        )

        pos_sim = self.sim(a_outputs, p_outputs)
        pos_exp = torch.exp(torch.div(pos_sim, self.temperature))

        neg_sim = self.sim(a_outputs, n_outputs)
        neg_exp = torch.exp(torch.div(neg_sim, self.temperature))

        log_probs = torch.log(pos_exp) - torch.log(pos_exp.sum() + neg_exp.sum())

        loss = -1 * log_probs

        return loss.mean()
