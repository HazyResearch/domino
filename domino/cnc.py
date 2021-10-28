import copy
from functools import partial

import meerkat as mk
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dosma import DicomReader
from PIL import Image
from tqdm import tqdm


def load_contrastive_dp(dp, num_a, num_p, num_n):
    """
    given a datapanel, create a new dp with
    (anchor,positive,negative) pairs for CnC
    using information from "group_id" column
    """

    # HACK: we are going to assume that:
    #       0: no pmx, no tube  --> majority group
    #       1: pmx, no tube     --> minority group
    #       2: no pmx, tube     --> minority group
    #       3: pmx, tube        --> majority group

    # notube_mask = np.logical_or(dp["group_id"].data == 0, dp["group_id"].data == 1)
    # tube_mask = np.logical_or(dp["group_id"].data == 2, dp["group_id"].data == 3)
    positive_entries_0 = dp[dp["group_id"].data == 2]
    negative_entries_0 = dp[dp["group_id"].data == 1]
    positive_entries_3 = dp[dp["group_id"].data == 1]
    negative_entries_3 = dp[dp["group_id"].data == 2]

    # filter out minorty classes since anchors are only from majority classes
    majority_mask = np.logical_or(dp["group_id"].data == 0, dp["group_id"].data == 3)
    contrastive_dp = dp[majority_mask]

    contastive_input_col = contrastive_dp[["group_id"]].to_lambda(
        fn=partial(
            contrastive_input_loader,
            positive_entries=(positive_entries_0, positive_entries_3),
            negative_entries=(negative_entries_0, negative_entries_3),
            num_p=num_p,
            num_n=num_n,
        )
    )

    contrastive_dp.add_column(
        "contrastive_input_pair",
        contastive_input_col,
        overwrite=True,
    )

    return contrastive_dp


def contrastive_input_loader(
    input_dict, positive_entries, negative_entries, num_p, num_n
):
    positive_entries_0, positive_entries_3 = positive_entries
    negative_entries_0, negative_entries_3 = negative_entries

    anchor_group_id = input_dict["group_id"]
    if anchor_group_id == 0:
        positive_idxs = np.random.choice(len(positive_entries_0), num_p)
        negative_idxs = np.random.choice(len(negative_entries_0), num_n)
        positive_inputs = positive_entries_0[positive_idxs]["input"].data
        negative_inputs = negative_entries_0[negative_idxs]["input"].data
        positive_targets = torch.Tensor(positive_entries_0[positive_idxs]["target"])
        negative_targets = torch.Tensor(negative_entries_0[negative_idxs]["target"])
    elif anchor_group_id == 3:
        positive_idxs = np.random.choice(len(positive_entries_3), num_p)
        negative_idxs = np.random.choice(len(negative_entries_3), num_n)
        positive_inputs = positive_entries_3[positive_idxs]["input"].data
        negative_inputs = negative_entries_3[negative_idxs]["input"].data
        positive_targets = torch.Tensor(positive_entries_3[positive_idxs]["target"])
        negative_targets = torch.Tensor(negative_entries_3[negative_idxs]["target"])

    # return (list(positive_entry), list(negative_entry))
    return (
        [positive_inputs, positive_targets],
        [negative_inputs, negative_targets],
    )


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

    def forward(self, contrastive_batch, model):

        # a_output, p_outputs, n_outputs = contrastive_batch
        a_output, p_outputs, n_outputs = (
            model(contrastive_batch[0].unsqueeze(0)).squeeze().unsqueeze(0),
            model(contrastive_batch[1]).squeeze(),
            model(contrastive_batch[2]).squeeze(),
        )

        pos_sim = self.sim(a_output, p_outputs)
        pos_exp = torch.exp(torch.div(pos_sim, self.temperature))
        pos_exp_sum = pos_exp.sum(0, keepdim=True)

        neg_sim = self.sim(a_output, n_outputs)
        neg_exp = torch.exp(torch.div(neg_sim, self.temperature))
        neg_exp_sum = neg_exp.sum(0, keepdim=True)

        log_probs = torch.log(pos_exp) - torch.log(pos_exp + neg_exp_sum)

        loss = -1 * log_probs

        return loss.mean(), pos_sim.mean(), neg_sim.mean()
