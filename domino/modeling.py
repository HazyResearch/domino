import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torchvision.models import DenseNet as _DenseNet
from torchvision.models import ResNet as _ResNet
from torchvision.models.densenet import _load_state_dict
from torchvision.models.densenet import model_urls as densenet_model_urls
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls as resnet_model_urls
from torchvision.models.utils import load_state_dict_from_url

from domino.transformer import TransformerEncoder, TransformerEncoderLayer


class ResNet(_ResNet):

    ACTIVATION_DIMS = [64, 128, 256, 512]
    ACTIVATION_WIDTH_HEIGHT = [64, 32, 16, 8]
    RESNET_TO_ARCH = {"resnet18": [2, 2, 2, 2], "resnet50": [3, 4, 6, 3]}

    def __init__(
        self,
        num_classes: int,
        arch: str = "resnet18",
        dropout: float = 0.0,
        pretrained: bool = True,
    ):
        if arch not in self.RESNET_TO_ARCH:
            raise ValueError(
                f"config['classifier'] must be one of: {self.RESNET_TO_ARCH.keys()}"
            )

        block = BasicBlock if arch == "resnet18" else Bottleneck
        super().__init__(block, self.RESNET_TO_ARCH[arch])
        if pretrained:
            state_dict = load_state_dict_from_url(
                resnet_model_urls[arch], progress=True
            )
            self.load_state_dict(state_dict)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(512 * block.expansion, num_classes)
        )


class DenseNet(_DenseNet):

    DENSENET_TO_ARCH = {
        "densenet121": {
            "growth_rate": 32,
            "block_config": (6, 12, 24, 16),
            "num_init_features": 64,
        }
    }

    def __init__(
        self, num_classes: int, arch: str = "densenet121", pretrained: bool = True
    ):
        if arch not in self.DENSENET_TO_ARCH:
            raise ValueError(
                f"config['classifier'] must be one of: {self.DENSENET_TO_ARCH.keys()}"
            )

        super().__init__(**self.DENSENET_TO_ARCH[arch])
        if pretrained:
            _load_state_dict(self, densenet_model_urls[arch], progress=True)

        self.classifier = nn.Linear(self.classifier.in_features, num_classes)


class SequenceModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        encoder,
        nheads=8,
        T=50,
        num_classes=2,
    ):
        super(SequenceModel, self).__init__()

        self.encoder_type = encoder
        self.T = T
        self.fc = nn.Linear(input_size, int(hidden_size / 2))

        if encoder == "lstm":
            self.encoder = nn.LSTM(
                input_size=int(hidden_size / 2),
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
            if bidirectional:
                num_feats = 2 * hidden_size
            else:
                num_feats = hidden_size
        elif encoder == "transformer":
            self.encoder = TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=int(hidden_size / 2),
                    nhead=nheads,
                    dim_feedforward=hidden_size,
                ),
                num_layers,
                LayerNorm(int(hidden_size / 2)),
            )
            num_feats = int(hidden_size / 2)

        elif encoder == "1dconv":
            if num_layers == 2:
                self.encoder = nn.Sequential(
                    nn.Conv1d(int(hidden_size / 2), hidden_size, 3),
                    nn.AvgPool1d(2),
                    nn.Conv1d(hidden_size, hidden_size, 2),
                    nn.AvgPool1d(2),
                )
            elif num_layers == 3:
                self.encoder = nn.Sequential(
                    nn.Conv1d(int(hidden_size / 2), hidden_size, 3),
                    nn.AvgPool1d(2),
                    nn.Conv1d(hidden_size, hidden_size, 2),
                    nn.AvgPool1d(2),
                    nn.Conv1d(hidden_size, hidden_size, 2),
                    nn.AvgPool1d(2),
                )

            num_feats = hidden_size

        self.classifier = nn.Linear(num_feats, num_classes)

    def forward(self, x_padded, seq_len):
        x_emb = self.fc(x_padded)
        if self.encoder_type == "lstm":
            x = torch.nn.utils.rnn.pack_padded_sequence(
                x_emb,
                seq_len.cpu().numpy(),
                batch_first=True,
                enforce_sorted=False,
            )
            output, _ = self.encoder(x)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        elif self.encoder_type == "1dconv":
            x = x_emb[:, : self.T, :]
            x = x.transpose(1, 2)
            output = self.encoder(x)
            output = output.transpose(1, 2)
        else:
            x = x_emb[:, : self.T, :]
            output = self.encoder(x)

        # return self.classifier(output[:, -1, :])
        return self.classifier(output.mean(1))
