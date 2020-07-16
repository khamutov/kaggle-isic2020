from pathlib import Path
from typing import Any, Union

import click
import torch


class RunOption:
    def __init__(self, name: str, default: Any, desc: str):
        self.name = name
        self.default = default
        self.desc = desc
        self.is_flag = False
        self.type = None
        self.callback = None

    def flag(self):
        self.is_flag = True
        return self

    def path(self):
        self.type = click.Path(exists=True)
        self.callback = RunOption._to_path
        return self

    @staticmethod
    def _to_path(_ctx, _param, value):
        if value is not None:
            return Path(value)


options = [
    RunOption(name="epochs", default=10, desc="Number of epoches for training."),
    RunOption(name="batch_size", default=64, desc="Train batch size."),
    RunOption(name="patience", default=3, desc="early stopping patience"),
    RunOption(name="lr_patience", default=1, desc="patience for learning rate"),
    RunOption(name="learning_rate", default=0.001, desc="Learning Rate"),
    RunOption(name="weight_decay", default=0.0, desc="Decay Factor"),
    RunOption(name="lr_factor", default=0.4, desc=""),
    RunOption(name="num_workers", default=0, desc="num epoches"),
    RunOption(name="dataset_malignant_256", default=None, desc="path to external malignant-256 dataset").path(),
    RunOption(name="dataset_official", default=None, desc="path to official ISIC dataset").path(),
    RunOption(name="mlflow_tracking_url", default=None, desc="mlflow tracking url"),
    RunOption(name="mlflow_experiment", default=None, desc="mlflow tracking url"),
    RunOption(name="device", default=None, desc="device: cpu, cuda, cuda:1"),
    RunOption(name="tta", default=11, desc="test time augmentation steps"),
    RunOption(name="dry_run", default=False,
              desc="run train on small set, do not track in MLflow. For sanity check only.").flag(),
]


class RunOptions:
    def __init__(self):
        # type-hint stubs
        self.epochs = None
        self.batch_size = None
        self.patience = None
        self.lr_patience = None
        self.learning_rate = None
        self.weight_decay = None
        self.lr_factor = None
        self.num_workers = None
        self.dataset_malignant_256: Union[None, Path] = None
        self.dataset_official: Union[None, Path] = None
        self.mlflow_tracking_url = None
        self.mlflow_experiment = None
        self._device = None
        self.tta = None
        self.dry_run = None
        for option in options:
            self.__setattr__(option.name, option.default)

    def is_track_mlflow(self):
        return self.mlflow_tracking_url is not None \
               and len(self.mlflow_tracking_url) > 0 \
               and not self.dry_run

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if value is None:
            value = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = value
