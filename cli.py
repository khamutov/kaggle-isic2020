import os
from pathlib import Path
from typing import Any, Union

import click
import mlflow
import torch
from tqdm import tqdm

from utils.seeds import seed_everything


class RunOption:
    def __init__(self, name: str, default: Any, desc: str):
        self.name = name
        self.default = default
        self.desc = desc
        self.is_flag = False
        self.type = None
        self.callback = None

    def flag(self):
        self.name = f"{self.name}/--no-{self.name}"
        self.is_flag = True
        return self

    def path(self):
        self.type = click.Path(exists=True)
        self.callback = RunOption._to_path
        return self

    def choice(self, values):
        self.type = click.Choice(values)
        return self

    def integer(self):
        self.type = int
        return self

    @staticmethod
    def _to_path(_ctx, _param, value):
        if value is not None:
            return Path(value)


OPTIM_ADAM = "Adam"
OPTIM_ADAMW = "AdamW"
OPTIM_SGD = "SGD"

SCHED_1CYC = "OneCycleLR"
SCHED_COSINE = "CosineWithWarmupLR"
SCHED_DEOTTE = "DeotteWithWarmupLR"

MODEL_EFFICIENTNET_B0 = "efficientnet-b0"
MODEL_EFFICIENTNET_B1 = "efficientnet-b1"
MODEL_EFFICIENTNET_B2 = "efficientnet-b2"
MODEL_EFFICIENTNET_B3 = "efficientnet-b3"
MODEL_EFFICIENTNET_B4 = "efficientnet-b4"
MODEL_EFFICIENTNET_B5 = "efficientnet-b5"
MODEL_EFFICIENTNET_B6 = "efficientnet-b6"
MODEL_EFFICIENTNET_B7 = "efficientnet-b7"
MODEL_RESNEST50 = "resnest50"
MODEL_EFFICIENTNET_B0_NS = "tf_efficientnet_b0_ns"
MODEL_EFFICIENTNET_B6_NS = "tf_efficientnet_b6_ns"

batches = {
    256: {
        MODEL_EFFICIENTNET_B0: 100,
        MODEL_EFFICIENTNET_B1: 1000,  # TODO: fill in
        MODEL_EFFICIENTNET_B2: 64,
        MODEL_EFFICIENTNET_B3: 1000,  # TODO: fill in
        MODEL_EFFICIENTNET_B4: 1000,  # TODO: fill in
        MODEL_EFFICIENTNET_B5: 15,
        MODEL_EFFICIENTNET_B6: 1000,  # TODO: fill in
        MODEL_EFFICIENTNET_B7: 1000,  # TODO: fill in
    },
    384: {
        MODEL_EFFICIENTNET_B0: 100,
        MODEL_EFFICIENTNET_B1: 1000,  # TODO: fill in
        MODEL_EFFICIENTNET_B2: 64,
        MODEL_EFFICIENTNET_B3: 1000,  # TODO: fill in
        MODEL_EFFICIENTNET_B4: 1000,  # TODO: fill in
        MODEL_EFFICIENTNET_B5: 15,
        MODEL_EFFICIENTNET_B6: 1000,  # TODO: fill in
        MODEL_EFFICIENTNET_B7: 1000,  # TODO: fill in
    },
}

OUTPUT_DEFAULT_PATH = "./"
options = [
    RunOption(name="epochs", default=10, desc="Number of epoches for training."),
    RunOption(name="batch_size", default=None, desc="Train batch size.").integer(),
    RunOption(name="patience", default=10, desc="early stopping patience"),
    RunOption(name="lr_patience", default=1, desc="patience for learning rate"),
    RunOption(name="learning_rate", default=0.000311878285355134, desc="Learning Rate"),
    RunOption(name="weight_decay", default=6.191958838098266e-05, desc="Decay Factor"),
    RunOption(name="lr_factor", default=0.4, desc=""),
    RunOption(
        name="loss_bce_label_smoothing",
        default=0.03,
        desc="Label smoothing for BCE loss",
    ),
    RunOption(name="num_workers", default=0, desc="num epoches"),
    RunOption(name="input_size", default="256", desc="input image sizes")
    .integer()
    .choice(["256", "384"]),
    RunOption(
        name="datasets_path", default=None, desc="path to datasets directory"
    ).path(),
    RunOption(
        name="dataset_official", default=None, desc="path to official ISIC dataset"
    ).path(),
    RunOption(name="mlflow_tracking_url", default=None, desc="mlflow tracking url"),
    RunOption(name="mlflow_experiment", default=None, desc="mlflow tracking url"),
    RunOption(name="device", default=None, desc="device: cpu, cuda, cuda:1"),
    RunOption(name="tta", default=15, desc="test time augmentation steps"),
    RunOption(
        name="dry_run",
        default=False,
        desc="run train on small set, do not track in MLflow. For sanity check only.",
    ).flag(),
    RunOption(
        name="no_cv",
        default=False,
        desc="Do not make cross-validation folds (for testing hypothesis).",
    ).flag(),
    RunOption(name="optim", default=OPTIM_ADAMW, desc="Optimizer").choice(
        [OPTIM_ADAM, OPTIM_ADAMW, OPTIM_SGD]
    ),
    RunOption(name="scheduler", default=SCHED_1CYC, desc="Scheduler").choice(
        [SCHED_1CYC, SCHED_COSINE, SCHED_DEOTTE]
    ),
    RunOption(name="model", default=MODEL_EFFICIENTNET_B0, desc="Model name").choice(
        [
            MODEL_EFFICIENTNET_B0,
            MODEL_EFFICIENTNET_B1,
            MODEL_EFFICIENTNET_B2,
            MODEL_EFFICIENTNET_B3,
            MODEL_EFFICIENTNET_B4,
            MODEL_EFFICIENTNET_B5,
            MODEL_EFFICIENTNET_B6,
            MODEL_EFFICIENTNET_B7,
            MODEL_RESNEST50,
            MODEL_EFFICIENTNET_B0_NS,
            MODEL_EFFICIENTNET_B6_NS,
        ]
    ),
    RunOption(
        name="advanced_hair_augmentation",
        default=False,
        desc="Augmentation AdvancedHairAugmentation",
    ).flag(),
    RunOption(
        name="jpeg_compression", default=True, desc="Augmentation JpegCompression"
    ).flag(),
    RunOption(name="rotate", default=True, desc="Augmentation Rotate").flag(),
    RunOption(
        name="optical_distortion", default=True, desc="Augmentation OpticalDistortion"
    ).flag(),
    RunOption(
        name="grid_distortion", default=True, desc="Augmentation GridDistortion"
    ).flag(),
    RunOption(
        name="piecewise_affine", default=False, desc="Augmentation IAAPiecewiseAffine"
    ).flag(),
    RunOption(
        name="horizontal_flip", default=True, desc="Augmentation HorizontalFlip"
    ).flag(),
    RunOption(
        name="vertical_flip", default=True, desc="Augmentation VerticalFlip"
    ).flag(),
    RunOption(
        name="gaussian_blur", default=False, desc="Augmentation GaussianBlur"
    ).flag(),
    RunOption(
        name="random_brightness_contrast",
        default=True,
        desc="Augmentation RandomBrightnessContrast",
    ).flag(),
    RunOption(
        name="hue_saturation_value",
        default=False,
        desc="Augmentation HueSaturationValue",
    ).flag(),
    RunOption(name="cutout", default=True, desc="Augmentation Cutout").flag(),
    RunOption(
        name="output_path",
        default=OUTPUT_DEFAULT_PATH,
        desc="Path to save trained models",
    ),
    RunOption(name="train", default=True, desc="Enable/disable train").flag(),
    RunOption(name="hpo", default=False, desc="Enable HPO").flag(),
    RunOption(name="hpo_n_trials", default=10, desc="Number trials for HPO"),
    RunOption(name="hpo_pruning", default=True, desc="Enable/disable pruning").flag(),
    RunOption(name="sched_warmup", default=0.1, desc="warmup percent of all epochs"),
    RunOption(name="alpha", default=0.7300593232355836, desc="focal loss alpha"),
    RunOption(
        name="new_external_dataset", default=False, desc="use new external dataset"
    ).flag(),
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
        self.loss_bce_label_smoothing = None
        self.num_workers = None
        self.input_size = None
        self.datasets_path: Union[None, Path] = None
        self.dataset_official: Union[None, Path] = None
        self.mlflow_tracking_url = None
        self.mlflow_experiment = None
        self._device = None
        self.tta = None
        self.dry_run = None
        self.no_cv = False
        self.optim = None
        self.scheduler = None
        self.model = None
        self.advanced_hair_augmentation = None
        self.jpeg_compression = None
        self.rotate = None
        self.optical_distortion = None
        self.grid_distortion = None
        self.piecewise_affine = None
        self.horizontal_flip = None
        self.vertical_flip = None
        self.gaussian_blur = None
        self.random_brightness_contrast = None
        self.hue_saturation_value = None
        self.cutout = None
        self.output_path = None
        self.train = None
        self.pos_weight = 10.0
        self.hpo = False
        self.hpo_n_trials = 0
        self.hpo_pruning = True
        self.sched_warmup = 0.1
        self.alpha = 0.75
        self.gamma = 2.0
        self.new_external_dataset = True
        for option in options:
            self.__setattr__(option.name.split("/")[0], option.default)

    def is_track_mlflow(self):
        return (
            self.mlflow_tracking_url is not None
            and len(self.mlflow_tracking_url) > 0
            and not self.dry_run
        )

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if value is None:
            value = 0 if torch.cuda.is_available() else "cpu"
        self._device = value

    def post_init(self):
        # batch size
        self.input_size = int(self.input_size)
        if self.batch_size is None:
            if self.input_size in batches:
                res_batches = batches.get(self.input_size)
                if self.model in res_batches:
                    self.batch_size = res_batches.get(self.model)
                else:
                    raise Exception(f"there is no default batch for model {self.model}")
            else:
                raise Exception(
                    f"there is no default batch for input size {self.input_size}"
                )

        tqdm.pandas()
        # warnings.filterwarnings("ignore")

        seed = 1234
        seed_everything(seed)

        # check device available
        # _tmp_tensor = torch.rand(1).to(self.device)
        # del _tmp_tensor

        torch.cuda.set_device(f"cuda:{self.device}")

        if self.is_track_mlflow():
            if not self.mlflow_experiment:
                raise Exception(
                    "mlflow experiment not set! Use option --mlflow_experiment"
                )

            mlflow.set_tracking_uri(self.mlflow_tracking_url)
            mlflow.set_experiment(self.mlflow_experiment)
            run_info = mlflow.start_run()

            # overrride output path if not set
            if self.output_path == OUTPUT_DEFAULT_PATH:
                self.output_path = str(
                    Path(os.getcwd()) / "models" / self.model / run_info.info.run_id
                )
            mlflow.log_params(self.__dict__)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        # try:
        #     set_start_method("spawn")
        # except RuntimeError:
        #     pass

    def dataset_2020(self):
        return self.datasets_path / f"jpeg-melanoma-{self.input_size}x{self.input_size}"

    def dataset_2019(self):
        return self.datasets_path / f"jpeg-isic2019-{self.input_size}x{self.input_size}"

    def dataset_external_v2(self):
        return self.datasets_path / f"malignant-v2-{self.input_size}x{self.input_size}"
