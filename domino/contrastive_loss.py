"""
Copied from https://github.com/HazyResearch/unagi/blob/main/src/unagi/tasks/loss_fns/contrastive_loss.py#L46
"""
import numpy as np
import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        type="l_spread",  # sup_con, sim_clr, l_attract, l_spread
        temp=0.5,
        pos_in_denom=False,  # as per dan, false by default
        log_first=True,  # TODO (ASN): should this be true (false originally)
        a_lc=1.0,
        a_spread=1.0,
        lc_norm=False,
        use_labels=True,
    ):
        super().__init__()
        self.temp = temp
        self.log_first = log_first
        self.a_lc = a_lc
        self.a_spread = a_spread
        self.pos_in_denom = pos_in_denom
        self.lc_norm = lc_norm
        self.use_labels = use_labels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if type == "sup_con":
            print(f"Using {type} contrastive loss function")
            self.a_spread = 0
            self.pos_in_denom = True  # working
        elif type == "l_attract":
            print(f"Using {type} contrastive loss function")
            self.a_spread = 0
            self.pos_in_denom = False  # working
        elif type == "l_repel":
            print(f"Using {type} contrastive loss function")
            self.a_spread = 1
            self.a_lc = 0
        elif type == "sim_clr":
            print(f"Using {type} contrastive loss function")
            self.a_spread = 0
            self.a_lc = 1
            self.use_labels = False

    def forward(self, x, labels):
        # x has shape batch * num views * dimension
        # labels has shape batch * num views
        b, nViews, d = x.size()
        vs = torch.split(x, 1, dim=1)  # images indexed by view
        if not self.use_labels:
            labels = torch.full(labels.shape, -1)
        ts = torch.split(labels, 1, dim=1)  # labels indexed by view
        l = torch.tensor(0.0).to(self.device)
        pairs = nViews * (nViews - 1) // 2

        for ii in range(nViews):
            vi = vs[ii].squeeze()
            ti = ts[ii].squeeze()

            ti_np = np.array([int(label) for label in ti])
            for jj in range(ii):
                vj = vs[jj].squeeze()

                # num[i,j] is f(xi) * f(xj) / tau, for i,j
                if self.lc_norm:
                    num = (
                        torch.einsum("b d, c d -> b c", vi, vj)
                        .div(self.temp)
                        .div(torch.norm(vi, dim=1) * torch.norm(vj, dim=1))
                    )
                else:
                    num = torch.einsum("b d, c d -> b c", vi, vj).div(self.temp)

                # store the first positive (augmentation of the same view)
                pos_ones = []
                neg_ones = []  # store the first negative
                M_indices = []
                div_factor = []

                for i, cls in enumerate(ti_np):
                    # fall back to SimCLR
                    pos_indices = torch.tensor([i]).to(self.device)
                    if cls != -1:
                        pos_indices = torch.where(ti == cls)[0]

                    # fall back to SimCLR
                    neg_indices = torch.tensor(
                        [idx for idx in range(ti.shape[0]) if idx != i]
                    ).to(self.device)

                    if cls != -1:
                        neg_indices = torch.where(ti != cls)[0]

                    all_indices = torch.stack(
                        [
                            torch.cat(
                                (
                                    pos_indices[j : j + 1],
                                    neg_indices,
                                )
                            )
                            for j in range(len(pos_indices))
                        ]
                    )

                    # store all the positive indices
                    pos_ones.append(pos_indices)

                    # store all the negative indices that go up to m
                    neg_ones.append(neg_indices)
                    M_indices.append(all_indices)
                    div_factor.append(len(pos_indices))

                # denominator for each point in the batch
                denominator = torch.stack(
                    [
                        # reshape num with an extra dimension, then take the
                        # sum over everything
                        torch.logsumexp(num[i][M_indices[i]], 1).sum()
                        for i in range(len(ti))
                    ]
                )

                # numerator
                numerator = torch.stack(
                    [
                        # sum over all the positives
                        torch.sum(-1 * num[i][pos_ones[i]])
                        #                     -1 * num[i][pos_ones[i]]
                        for i in range(len(ti))
                    ]
                )

                log_prob = numerator + denominator

                if self.a_spread > 0.0:
                    assert self.a_lc + self.a_spread != 0

                    numerator_spread = -1 * torch.diagonal(num, 0)
                    denominator_spread = torch.stack(
                        [
                            # reshape num with an extra dimension,
                            # then take the sum over everything
                            torch.logsumexp(num[i][pos_ones[i]], 0).sum()
                            for i in range(len(ti))
                        ]
                    )
                    log_prob_spread = numerator_spread + denominator_spread

                    a = (
                        self.a_lc
                        * log_prob.div(torch.tensor(div_factor).to(self.device))
                        + self.a_spread * log_prob_spread
                    ) / (self.a_lc + self.a_spread)
                else:
                    a = self.a_lc * log_prob.div(
                        torch.tensor(div_factor).to(self.device)
                    )

                l += a.mean()

        out = l / pairs
        return out
