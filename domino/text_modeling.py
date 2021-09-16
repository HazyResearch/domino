from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, BertConfig, BertModel, BertTokenizer


class bert_labeler(nn.Module):
    def __init__(self):
        super(bert_labeler, self).__init__()
        config = BertConfig.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            config=config,
            cache_dir="/media/nvme_data/pretrained_models/cache",
        )
        self.hidden_size = self.bert.pooler.dense.in_features

    def forward(self, source_padded, attention_mask):
        output = self.bert(source_padded, attention_mask=attention_mask)
        final_hidden = output[0]
        hidden_states = output[2][1:]
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        return cls_hidden, hidden_states


class CheXbert(nn.Module):
    def __init__(self, chexbert_pth):
        super(CheXbert, self).__init__()
        model = bert_labeler()
        model_param_names = [name for name, _ in model.named_parameters()]

        state_dict = torch.load(chexbert_pth)["model_state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            if name not in model_param_names:
                print("CheXbert: skipping param", name)
                continue
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        self.model = model
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        #    if '10' in name or '11' in name or 'pooler' in name:
        #         param.requires_grad = True

        self.output = nn.Sequential(
            nn.Linear(self.model.hidden_size, self.model.hidden_size), nn.Tanh()
        )

    def forward(self, batch, attn_mask, multi=None, clip=False):
        out, hidden_states = self.model(batch, attn_mask)
        if clip:
            return out

        out = self.output(out)
        if multi != None:
            layer_out = {}
            for layer in multi:
                layer_out[layer] = hidden_states[layer][:, 0, :]
            return out, layer_out
        else:
            return out, None


class TextCheXbert(nn.Module):
    def __init__(self, chexbert_pth, projection_dim: None):
        super(TextCheXbert, self).__init__()
        self.chexbert = CheXbert(chexbert_pth)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        if projection_dim:
            in_feats = self.chexbert.output[0].in_features
            self.chexbert.output[0] = nn.Linear(in_feats, projection_dim)

    def forward(self, report):
        inp = self.tokenizer(
            report,
            return_tensors="pt",
            padding=True,
            max_length=self.tokenizer.model_max_length,
            truncation_strategy="only_first",
        )
        chex_vector, _ = self.chexbert(inp.input_ids.cuda(), inp.attention_mask.cuda())
        return chex_vector
