from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torchvision
from datasets import DatasetInfo, NamedSplit
from robustnessgym.core.identifier import Identifier
from torchvision import transforms
from wilds import get_dataset
from wilds.datasets.fmow_dataset import FMoWDataset


class WildsDataPane(DataPane):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        version: str = None,
        identifier: Identifier = None,
        column_names: List[str] = None,
        info: DatasetInfo = None,
        split: Optional[NamedSplit] = None,
    ):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        input_column = WildsInputColumn(
            dataset_name=dataset_name, version=version, root_dir=root_dir
        )
        output_column = NumpyArrayColumn(data=input_column.data.y_array)
        metadata_column = NumpyArrayColumn(data=input_column.data.metadata_array)
        super(WildsDataPane, self).__init__(
            {"input": input_column, "output": output_column, "meta": metadata_column},
            identifier=identifier,
            column_names=column_names,
            info=info,
            split=split,
        )


class WildsInputColumn(AbstractColumn):
    def __init__(
        self, dataset_name: str, version: str = None, root_dir=None, **dataset_kwargs
    ):
        self._state = {
            "dataset_name": dataset_name,
            "version": version,
            "root_dir": root_dir,
            "dataset_kwargs": dataset_kwargs,
        }
        self.dataset_name = dataset_name
        self.version = version
        self.root_dir = root_dir
        self.dataset_kwargs = dataset_kwargs
        dataset = get_dataset(
            dataset=dataset_name, version=version, root_dir=root_dir, **dataset_kwargs
        )
        super(WildsInputColumn, self).__init__(data=dataset, collate_fn=dataset.collate)

    def _get_cell(self, index: int):
        return self.data.get_input(index)

    @property
    def write_data(self):
        return self._write_data

    def get_state(self) -> Dict:
        return self._state

    @classmethod
    def from_state(cls, state: Dict) -> WildsInputColumn:
        return cls(**state)


FMOW_CONFIG = {
    "model_name": "wilds_model",
    "dataset": "fmow",
    "split_scheme": "official",
    "dataset_kwargs": {"oracle_training_set": False, "seed": 111, "use_ood_val": True},
    "metrics": ["accuracy"],
    "model_type": "densenet121",
    "model_kwargs": {"pretrained": True},
    "train_transform": "image_base",
    "eval_transform": "image_base",
    "loss_function": "cross_entropy",
    "groupby_fields": [
        "year",
    ],
    "val_metric": "acc_worst_region",
    "val_metric_decreasing": False,
    "optimizer": "Adam",
    "scheduler": "StepLR",
    "scheduler_kwargs": {"gamma": 0.96},
    "batch_size": 64,
    "lr": 0.0001,
    "weight_decay": 0.0,
    "n_epochs": 50,
    "n_groups_per_batch": 8,
    "irm_lambda": 1.0,
    "coral_penalty_weight": 0.1,
    "algo_log_metric": "accuracy",
    "process_outputs_function": "multiclass_logits_to_pred",
    "num_classes": 62,
    "target_resolution": (224, 224),
    "model_path": "/home/common/datasets/fmow_v1.1/models/fmow_erm_ID_seed0/best_model.pth",
}


def fmow_transform(img):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img)


def initialize_image_base_transform(config, dataset):
    transform_steps = []
    if dataset.original_resolution is not None and min(
        dataset.original_resolution
    ) != max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))
    if config["target_resolution"] is not None and config["dataset"] != "fmow":
        transform_steps.append(transforms.Resize(config["target_resolution"]))
    transform_steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    transform = transforms.Compose(transform_steps)
    return transform


def get_wilds_model(
    model_type: str = "resnet50",
    model_path: str = None,
    d_out: int = 2,
    is_featurizer: bool = False,
    model_kwargs: dict = None,
):
    model_kwargs = {} if model_kwargs is None else model_kwargs
    if model_type in ("resnet50", "resnet34", "wideresnet50", "densenet121"):
        if is_featurizer:
            featurizer = initialize_torchvision_model(
                name=model_type, d_out=None, **model_kwargs
            )
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_torchvision_model(
                name=model_type, d_out=d_out, **model_kwargs
            )
    else:
        raise ValueError(f"Model: {model_type} not recognized.")

    if model_path is not None:
        print(f"loading weights from {model_path}...")
        state_dict = torch.load(model_path)

        # wrap the model so it resembles the wilds setup
        class WrapperModule(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

        _model = WrapperModule(model)
        _model.load_state_dict(state_dict["algorithm"])
        return _model.model

    return model


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self, d):
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x):
        return x


def initialize_torchvision_model(name, d_out, **kwargs):
    # get constructor and last layer names
    if name == "wideresnet50":
        constructor_name = "wide_resnet50_2"
        last_layer_name = "fc"
    elif name == "densenet121":
        constructor_name = name
        last_layer_name = "classifier"
    elif name in ("resnet50", "resnet34"):
        constructor_name = name
        last_layer_name = "fc"
    else:
        raise ValueError(f"Torchvision model {name} not recognized")
    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        model.d_out = d_features
    else:  # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out

    setattr(model, last_layer_name, last_layer)
    return model
