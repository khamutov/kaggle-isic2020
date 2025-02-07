import glob
import math
import os
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path

import click
import configobj
import cv2
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import timm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, Style
from efficientnet_pytorch import EfficientNet
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.classification import AUROC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange

import cli
from augmentation.hairs import AdvancedHairAugmentation
from loss import sigmoid_focal_loss
from utils.seeds import seed_everything

try:
    import mlflow
except ImportError:
    print("Run without mlflow")


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2


FOLDS = 5


def get_tta_transforms(config):
    return A.Compose(
        [
            A.NoOp(),
            A.NoOp(),
            A.Rotate(limit=80, p=1.0),
            A.OneOf([A.NoOp(), A.NoOp(), A.NoOp()]),
            # A.RandomSizedCrop(min_max_height=(int(resolution*0.7), input_res),
            #                   height=resolution, width=resolution, p=1.0),
            A.NoOp(),
            A.NoOp(),
            A.GaussianBlur(p=0.3) if config.gaussian_blur else A.NoOp(),
            A.OneOf([A.NoOp(), A.HueSaturationValue(hue_shift_limit=0)]),
            A.NoOp(),
            A.Normalize(),
            ToTensorV2(),
        ],
        p=1.0,
    )


def get_train_transforms(config):
    return A.Compose(
        [
            AdvancedHairAugmentation(
                hairs_folder="/home/a.khamutov/kaggle-datasource/melanoma-hairs"
            )
            if config.advanced_hair_augmentation
            else A.NoOp(),
            A.JpegCompression(p=0.5) if config.jpeg_compression else A.NoOp(),
            A.Rotate(limit=80, p=1.0) if config.rotate else A.NoOp(),
            A.OneOf(
                [
                    A.OpticalDistortion() if config.optical_distortion else A.NoOp(),
                    A.GridDistortion() if config.grid_distortion else A.NoOp(),
                    A.IAAPiecewiseAffine() if config.piecewise_affine else A.NoOp(),
                ]
            ),
            # A.RandomSizedCrop(min_max_height=(int(resolution*0.7), input_res),
            #                   height=resolution, width=resolution, p=1.0),
            A.HorizontalFlip(p=0.5) if config.horizontal_flip else A.NoOp(),
            A.VerticalFlip(p=0.5) if config.vertical_flip else A.NoOp(),
            A.GaussianBlur(p=0.3) if config.gaussian_blur else A.NoOp(),
            A.OneOf(
                [
                    A.RandomBrightnessContrast()
                    if config.random_brightness_contrast
                    else A.NoOp(),
                    A.HueSaturationValue(hue_shift_limit=0)
                    if config.hue_saturation_value
                    else A.NoOp(),
                ]
            ),
            A.Cutout(
                num_holes=8,
                max_h_size=config.input_size // 5,
                max_w_size=config.input_size // 5,
                fill_value=0,
                p=0.75,
            )
            if config.cutout
            else A.NoOp(),
            A.Normalize(),
            ToTensorV2(),
        ],
        p=1.0,
    )


# def signal_handler(_sig, _frame):
#     print('You pressed Ctrl+C!')
#     sys.exit(0)
#
#
# signal.signal(signal.SIGINT, signal_handler)


# from huggingface transformers
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# This is a common train schedule for transfer learning.
# The learning rate starts near zero, then increases to a maximum, then decays over time.
def get_exp_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    steps_per_epoch: int,
    num_sustain_steps: int = 0,
    lr_decay: float = 0.8,
):
    def lrfn(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + num_sustain_steps:
            return 1.0

        return lr_decay ** (
            float(current_step - num_warmup_steps - num_sustain_steps) / steps_per_epoch
        )

    return LambdaLR(optimizer, lrfn)


class MelanomaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        imfolder: str,
        is_train: bool = True,
        transforms=None,
        meta_features=None,
    ):

        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.is_train = is_train
        self.meta_features = meta_features

    def __getitem__(self, index):
        im_path = os.path.join(
            self.imfolder, self.df.iloc[index]["image_name"] + ".jpg"
        )
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        metadata = np.array(
            self.df.iloc[index][self.meta_features].values, dtype=np.float32
        )

        if self.transforms:
            sample = self.transforms(image=image)
            image = sample["image"]

        if self.is_train:
            y = self.df.iloc[index]["target"]
            #             image = image.cuda()
            return (image, metadata), y
        else:
            return (image, metadata), -1  # const for common contract

    def __len__(self):
        return len(self.df)


def load_dataset(config: cli.RunOptions):
    train_df = pd.read_csv(config.dataset_2020() / "train.csv")
    train_df_2019 = pd.read_csv(config.dataset_2019() / "train.csv")
    train_df_external = pd.read_csv(config.dataset_external_v2() / "train_malig_2.csv")

    train_df["fold"] = train_df["tfrecord"]
    train_df_2019["fold"] = train_df_2019["tfrecord"]
    test_df = pd.read_csv(config.dataset_official / "test.csv")

    train_df["sex"] = train_df["sex"].map({"male": 1, "female": 0})
    train_df_2019["sex"] = train_df_2019["sex"].map({"male": 1, "female": 0})
    train_df_external["sex"] = train_df_external["sex"].map({"male": 1, "female": 0})

    test_df["sex"] = test_df["sex"].map({"male": 1, "female": 0})

    train_df["sex"] = train_df["sex"].fillna(-1)
    train_df_2019["sex"] = train_df_2019["sex"].fillna(-1)
    train_df_external["sex"] = train_df_external["sex"].fillna(-1)

    test_df["sex"] = test_df["sex"].fillna(-1)

    # imputing
    imp_mean = (train_df["age_approx"].sum()) / (
        train_df["age_approx"].count() - train_df["age_approx"].isna().sum()
    )
    train_df["age_approx"] = train_df["age_approx"].fillna(imp_mean)
    train_df_2019["age_approx"] = train_df_2019["age_approx"].fillna(imp_mean)
    train_df_external["age_approx"] = train_df_external["age_approx"].fillna(imp_mean)

    imp_mean_test = (test_df["age_approx"].sum()) / (test_df["age_approx"].count())
    test_df["age_approx"] = test_df["age_approx"].fillna(imp_mean_test)

    train_df["patient_id"] = train_df["patient_id"].fillna(0)
    train_df_2019["patient_id"] = train_df_2019["patient_id"].fillna(0)
    train_df_external["patient_id"] = train_df_external["patient_id"].fillna(0)

    # OHE
    # TODO: make on sklearn

    concat = pd.concat(
        [
            train_df["anatom_site_general_challenge"],
            train_df_2019["anatom_site_general_challenge"],
            train_df_external["anatom_site_general_challenge"],
            test_df["anatom_site_general_challenge"],
        ],
        ignore_index=True,
    )
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix="site")
    train_df = pd.concat([train_df, dummies.iloc[: train_df.shape[0]]], axis=1)
    train_df_2019 = pd.concat(
        [
            train_df_2019,
            dummies.iloc[
                train_df.shape[0] : train_df.shape[0] + train_df_2019.shape[0]
            ].reset_index(drop=True),
        ],
        axis=1,
    )
    train_df_external = pd.concat(
        [
            train_df_external,
            dummies.iloc[
                train_df.shape[0]
                + train_df_2019.shape[0] : train_df.shape[0]
                + train_df_2019.shape[0]
                + train_df_external.shape[0]
            ].reset_index(drop=True),
        ],
        axis=1,
    )
    test_df = pd.concat(
        [
            test_df,
            dummies.iloc[
                train_df.shape[0]
                + train_df_2019.shape[0]
                + train_df_external.shape[0] :
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    meta_features = ["sex", "age_approx"] + [
        col for col in train_df.columns if "site_" in col
    ]
    meta_features.remove("anatom_site_general_challenge")

    test_df = test_df.drop(["anatom_site_general_challenge"], axis=1)
    train_df = train_df.drop(["anatom_site_general_challenge"], axis=1)
    train_df_2019 = train_df_2019.drop(["anatom_site_general_challenge"], axis=1)
    train_df_external = train_df_external.drop(
        ["anatom_site_general_challenge"], axis=1
    )

    return train_df, train_df_2019, train_df_external, test_df, meta_features


@dataclass
class TrainResult:
    best_val: float
    pred = None
    pred_tta = None
    target = None
    names = None

    def __init__(self):
        pass


def find_checkpoint(fold_idx: int, config: cli.RunOptions):
    pattern = str(Path(config.output_path) / f"model_fold={fold_idx}_epoch=*.ckpt")
    matched_checkpoints = glob.glob(pattern)

    return matched_checkpoints


def assert_no_checkpoint_exists(fold_idx: int, config: cli.RunOptions):
    matched_checkpoints = find_checkpoint(fold_idx, config)
    if len(matched_checkpoints) > 0:
        print(
            Fore.RED,
            f"There is already exist a checkpoint for fold {fold_idx} in path {config.output_path}",
            Style.RESET_ALL,
        )
        raise Exception("output path already has model!")


def load_checkpoint(fold_idx: int, config: cli.RunOptions):
    matched_checkpoints = find_checkpoint(fold_idx, config)

    if len(matched_checkpoints) != 1:
        print(
            Fore.RED, f"Couldn't find best model for fold {fold_idx}", Style.RESET_ALL,
        )
        print(
            Fore.RED,
            f"Found {len(matched_checkpoints)} files for this fold",
            Style.RESET_ALL,
        )
        raise Exception("cannot load model from checkpoint")
    return IsicModel.load_from_checkpoint(checkpoint_path=matched_checkpoints[0])


def train_fit(
    train_df,
    train_df_2018,
    train_df_external,
    val_df,
    train_transform,
    tta_transform,
    test_transform,
    meta_features,
    config: cli.RunOptions,
    fold_idx: int,
    trial: optuna.trial.Trial,
) -> TrainResult:
    output_size = 1  # statics

    train_result = TrainResult()
    train_result.names = val_df["image_name"].to_numpy()

    model_path = (
        Path(config.output_path) / f"model_fold={fold_idx}_epoch={{epoch}}.ckpt"
    )

    train_dataset_2020 = MelanomaDataset(
        df=train_df,
        imfolder=config.dataset_2020() / "train",
        is_train=True,
        transforms=train_transform,
        meta_features=meta_features,
    )
    train_dataset_2018 = MelanomaDataset(
        df=train_df_2018,
        imfolder=config.dataset_2019() / "train",
        is_train=True,
        transforms=train_transform,
        meta_features=meta_features,
    )
    train_dataset_external = MelanomaDataset(
        df=train_df_external,
        imfolder=config.dataset_external_v2() / f"jpeg{config.input_size}",
        is_train=True,
        transforms=train_transform,
        meta_features=meta_features,
    )
    train_dataset_list = [train_dataset_2020, train_dataset_2018]
    if config.new_external_dataset:
        train_dataset_list.append(train_dataset_external)
    train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
    val = MelanomaDataset(
        df=val_df,
        imfolder=config.dataset_2020() / "train",
        is_train=True,
        transforms=test_transform,
        meta_features=meta_features,
    )
    val_tta = MelanomaDataset(
        df=val_df,
        imfolder=config.dataset_2020() / "train",
        is_train=True,
        transforms=tta_transform,
        meta_features=meta_features,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_tta_loader = DataLoader(
        dataset=val_tta,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    os.makedirs(Path(config.output_path) / f"{config.model}", exist_ok=True)
    tb_logger = TensorBoardLogger(
        save_dir=config.output_path, name=f"{config.model}", version=f"fold_{fold_idx}"
    )

    # find model checkpoint for this fold
    if config.train:
        assert_no_checkpoint_exists(fold_idx=fold_idx, config=config)

        if config.is_track_mlflow():
            run_id = mlflow.active_run().info.run_id
        else:
            run_id = str(uuid.uuid4())

        model = IsicModel(
            output_size=output_size,
            no_columns=len(meta_features),
            config=config,
            steps_per_epoch=int(len(train_dataset) / config.batch_size),
            run_id=run_id,
        )
    else:
        model = load_checkpoint(fold_idx=fold_idx, config=config)
        run_id = model.run_id  # extract run_id for predictions

    if config.hpo:
        # Filenames for each trial must be made unique in order to access each checkpoint.
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            Path(config.output_path) / f"trial_{trial.number}_{{epoch}}",
            monitor="val_auc",
            mode="max",
        )
        early_stop_callback = PyTorchLightningPruningCallback(trial, monitor="val_auc")
    else:
        early_stop_callback = False
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            model_path, save_top_k=1, monitor="val_auc", mode="max"
        )

    trainer = pl.Trainer(
        logger=tb_logger,
        tpu_cores=8 if "TPU_NAME" in os.environ.keys() else None,
        gpus=config.device,
        precision=16 if config.device else 32,
        max_epochs=config.epochs,
        # distributed_backend='ddp',
        benchmark=False,
        deterministic=False,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
    )
    if config.train:
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    # build OOF metrics
    if config.train:
        trainer.test(test_dataloaders=val_loader, ckpt_path="best")
    else:
        trainer.test(model=model, test_dataloaders=val_loader)

    fold_preds = torch.load(f"preds_{run_id}.pt")
    os.remove(f"preds_{run_id}.pt")
    best_auc = (
        roc_auc_score(val_df["target"].values, fold_preds.cpu().numpy())
        if val_df["target"].mean() > 0
        else 0.5
    )  # skip sanity check
    train_result.pred = fold_preds.cpu().numpy()
    train_result.target = val_df["target"].values

    # do not make tta on HPO
    if not config.hpo:
        fold_preds_tta = torch.zeros((len(val_tta))).type_as(fold_preds)
        for _ in trange(config.tta, desc="val TTA", leave=False):
            if config.train:
                trainer.test(test_dataloaders=val_tta_loader, ckpt_path="best")
            else:
                trainer.test(model=model, test_dataloaders=val_tta_loader)

            fold_preds_tta += torch.load(f"preds_{run_id}.pt")
            os.remove(f"preds_{run_id}.pt")
        fold_preds_tta /= config.tta
        train_result.pred_tta = fold_preds_tta.cpu().numpy()

    train_result.best_val = best_auc

    return train_result


def train_model_cv(
    train_df,
    train_df_2018,
    train_df_external,
    meta_features,
    config: cli.RunOptions,
    train_transform,
    tta_transform,
    test_transform,
    trial: optuna.trial.Trial,
):
    oof_pred = []
    oof_pred_tta = []
    oof_target = []
    oof_folds = []
    oof_names = []

    TRAIN_GROUPS = 15
    if config.no_cv:
        train_val_split_at = int(0.8 * TRAIN_GROUPS)
        splits = [
            (np.arange(train_val_split_at), np.arange(train_val_split_at, TRAIN_GROUPS))
        ]
    else:
        skf = KFold(n_splits=FOLDS, shuffle=True, random_state=47)
        splits = skf.split(np.arange(TRAIN_GROUPS))

    for fold_idx, (idxT, idxV) in enumerate(splits, 1):
        train_idx = train_df.loc[train_df["fold"].isin(idxT)].index
        train_idx_2018 = train_df_2018.loc[
            train_df_2018["fold"].isin(np.concatenate([idxT, idxV]) * 2)
        ].index
        val_idx = train_df.loc[train_df["fold"].isin(idxV)].index

        oof_names.append(train_df.iloc[val_idx]["image_name"].to_numpy())

        if config.is_track_mlflow():
            mlflow.start_run(nested=True, run_name="Fold {}".format(fold_idx))
            mlflow.log_param("fold", fold_idx)

        print(
            Fore.CYAN,
            "-" * 20,
            Style.RESET_ALL,
            Fore.MAGENTA,
            "Fold",
            fold_idx,
            Style.RESET_ALL,
            Fore.CYAN,
            "-" * 20,
            Style.RESET_ALL,
        )

        train_fit_df = train_df.iloc[train_idx].reset_index(drop=True)
        train_fit_df_2018 = train_df_2018.iloc[train_idx_2018].reset_index(drop=True)
        train_df_external = train_df_external.reset_index(drop=True)
        val_fit_df = train_df.iloc[val_idx].reset_index(drop=True)

        train_result = train_fit(
            train_df=train_fit_df,
            train_df_2018=train_fit_df_2018,
            train_df_external=train_df_external,
            val_df=val_fit_df,
            train_transform=train_transform,
            tta_transform=tta_transform,
            test_transform=test_transform,
            meta_features=meta_features,
            config=config,
            fold_idx=fold_idx,
            trial=trial,
        )

        oof_pred.append(train_result.pred)
        oof_pred_tta.append(train_result.pred_tta)
        oof_target.append(train_result.target)
        oof_folds.append(np.ones_like(oof_target[-1], dtype="int8") * fold_idx)

        if config.is_track_mlflow():
            if train_result.best_val:
                mlflow.log_metric("best_roc_auc", train_result.best_val)
            mlflow.end_run()

        if config.hpo:
            return train_result.best_val

    oof = np.concatenate(oof_pred).squeeze()
    oof_tta = np.concatenate(oof_pred_tta).squeeze()
    true = np.concatenate(oof_target)
    folds = np.concatenate(oof_folds)
    auc = roc_auc_score(true, oof) if true.mean() > 0 else 0.5
    auc_tta = roc_auc_score(true, oof_tta) if true.mean() > 0 else 0.5
    names = np.concatenate(oof_names)

    print(Fore.CYAN, "-" * 60, Style.RESET_ALL)
    print(Fore.MAGENTA, "OOF ROC AUC", auc, Style.RESET_ALL)
    print(Fore.MAGENTA, "OOF ROC AUC TTA", auc_tta, Style.RESET_ALL)
    print(Fore.CYAN, "-" * 60, Style.RESET_ALL)

    if config.is_track_mlflow():
        mlflow.log_metric("oof_roc_auc", auc)
        mlflow.log_metric("oof_roc_auc_tta", auc_tta)

    # SAVE OOF TO DISK
    df_oof = pd.DataFrame(dict(image_name=names, target=true, pred=oof, fold=folds))
    df_oof.to_csv(Path(config.output_path) / "oof.csv", index=False)

    df_oof_tta = pd.DataFrame(
        dict(image_name=names, target=true, pred=oof_tta, fold=folds)
    )
    df_oof_tta.to_csv(Path(config.output_path) / "oof_tta.csv", index=False)


def predict_model(
    test_df,
    meta_features,
    config: cli.RunOptions,
    train_transform,
    tta_transform,
    test_transform,
):
    print(Fore.MAGENTA, "Run prediction", Style.RESET_ALL)

    test = MelanomaDataset(
        df=test_df,
        imfolder=config.dataset_2020() / "test",
        is_train=False,
        transforms=tta_transform,
        meta_features=meta_features,
    )
    test_loader = DataLoader(
        dataset=test,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    preds = torch.zeros(
        (len(test)), dtype=torch.float32, device=f"cuda:{config.device}"
    )
    for fold_idx in trange(1, FOLDS + 1, desc="Fold"):

        model = load_checkpoint(fold_idx=fold_idx, config=config)
        run_id = model.run_id

        trainer = pl.Trainer(
            tpu_cores=8 if "TPU_NAME" in os.environ.keys() else None,
            gpus=config.device,
            precision=16 if config.device else 32,
            max_epochs=config.epochs,
            # distributed_backend='ddp',
            benchmark=False,
            deterministic=False,
        )

        fold_preds = torch.zeros((len(test))).type_as(preds)
        for _ in trange(config.tta, desc="TTA", leave=False):
            trainer.test(model=model, test_dataloaders=test_loader)

            fold_preds += torch.load(f"preds_{run_id}.pt")
            os.remove(f"preds_{run_id}.pt")
        fold_preds /= config.tta
        preds += fold_preds

    preds /= FOLDS

    submission = pd.DataFrame(
        dict(image_name=test_df["image_name"].to_numpy(), target=preds.cpu().numpy(),)
    )
    submission = submission.sort_values("image_name")
    submission_filename = Path(config.output_path) / "submission.csv"
    submission.to_csv(submission_filename, index=False)

    print(Fore.MAGENTA, f"saved to {submission_filename}", Style.RESET_ALL)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class IsicModel(pl.LightningModule):
    def __init__(
        self,
        *,
        output_size: int,
        no_columns: int,
        config: cli.RunOptions,
        steps_per_epoch: int,
        run_id: str,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = config
        self.run_id = run_id

        self.steps_per_epoch = steps_per_epoch

        self.no_columns = no_columns

        if config.model.startswith("efficientnet"):
            self.features = EfficientNet.from_pretrained(config.model)
            self.eff_net_out_features = getattr(self.features, "_fc").in_features
        elif config.model.startswith("tf_efficientnet"):
            self.features = timm.create_model(config.model, pretrained=True)
            self.eff_net_out_features = getattr(self.features, "classifier").in_features
        elif config.model == "resnest50":
            torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)
            # load pretrained models, using ResNeSt-50 as an example
            self.features = torch.hub.load(
                "zhanghang1989/ResNeSt", "resnest50", pretrained=True
            )
            self.eff_net_out_features = getattr(self.features, "fc").in_features
            self.features.fc = Identity()

        # (CSV) or Meta Features
        meta_features_out = 250
        self.csv = nn.Sequential(
            nn.Linear(self.no_columns, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(250, meta_features_out),
            nn.BatchNorm1d(meta_features_out),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        fc_hidden_size = 250
        self.classification = nn.Sequential(
            nn.Linear(self.eff_net_out_features + meta_features_out, fc_hidden_size),
            nn.Linear(fc_hidden_size, output_size),
        )

    def forward(self, image, csv_data):
        # IMAGE CNN
        if self.config.model.startswith("efficientnet"):
            image = self.features.extract_features(image)
        elif self.config.model.startswith("tf_efficientnet"):
            image = self.features.forward_features(image)
        elif self.config.model == "resnest50":
            image = self.features(image)

        # image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, self.eff_net_out_features)
        if "efficientnet" in self.config.model:
            features = F.adaptive_avg_pool2d(image, 1)
        else:
            features = image

        image = features.view(features.size(0), -1)

        # CSV FNN
        csv_data = self.csv(csv_data)

        # Concatenate
        image_csv_data = torch.cat((image, csv_data), dim=1)

        # CLASSIF
        out = self.classification(image_csv_data)

        return out

    def label_smoothing(self, y):
        return (
            y.float() * (1 - self.config.loss_bce_label_smoothing)
            + 0.5 * self.config.loss_bce_label_smoothing
        )

    def step(self, batch):
        data, y = batch

        y_hat = self(data[0], data[1]).flatten()
        y_smooth = self.label_smoothing(y)

        # CB_loss(y_hat, y, [10000, 1], 2, "focal", 0.9999, 2.0)

        loss = sigmoid_focal_loss(
            y_hat, y_smooth, self.config.alpha, self.config.gamma, reduction="mean"
        )
        # loss = focal_loss(y_smooth, y_hat, 0.25, 2.0)
        # loss = F.binary_cross_entropy_with_logits(
        #     y_hat, y_smooth, pos_weight=torch.tensor(self.config.pos_weight)
        # )

        return loss, y, y_hat.sigmoid()

    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        acc = (y_hat.round() == y).float().mean().item()
        tensorboard_logs = {"train_loss": loss, "acc": acc}

        return {"loss": loss, "acc": acc, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, y, y_hat = self.step(batch)
        return {"val_loss": loss, "y": y.detach(), "y_hat": y_hat.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs])
        y_hat = torch.cat([x["y_hat"] for x in outputs])
        auc = (
            AUROC()(pred=y_hat, target=y) if y.float().mean() > 0 else 0.5
        )  # skip sanity check
        acc = (y_hat.round() == y).float().mean().item()
        print(f"Epoch {self.current_epoch} val_loss:{avg_loss} auc:{auc}")
        tensorboard_logs = {"val_loss": avg_loss, "val_auc": auc, "val_acc": acc}
        return {
            "avg_val_loss": avg_loss,
            "val_auc": auc,
            "val_acc": acc,
            "log": tensorboard_logs,
            "progress_bar": {"val_loss": avg_loss},
        }

    def test_step(self, batch, batch_nb):
        data, _ = batch
        y_hat = self(data[0], data[1]).flatten().sigmoid()
        return {"y_hat": y_hat}

    def test_epoch_end(self, outputs):
        y_hat = torch.cat([x["y_hat"] for x in outputs])
        torch.save(y_hat, f"preds_{self.run_id}.pt")

    def configure_optimizers(self):
        if self.config.optim == cli.OPTIM_ADAM:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optim == cli.OPTIM_ADAMW:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optim == cli.OPTIM_SGD:
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise Exception(f"unknown optimizer f{self.config.optim}")

        if self.config.scheduler == cli.SCHED_1CYC:
            scheduler_instance = torch.optim.lr_scheduler.OneCycleLR(
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                optimizer=optimizer,
                steps_per_epoch=self.steps_per_epoch,
                pct_start=self.config.sched_warmup,
                div_factor=10,
                final_div_factor=100,
                base_momentum=0.90,
                max_momentum=0.95,
            )
            scheduler = {
                "scheduler": scheduler_instance,
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 1,
            }
        elif self.config.scheduler == cli.SCHED_COSINE:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=3 * self.steps_per_epoch,
                num_training_steps=self.config.epochs * self.steps_per_epoch,
            )
        elif self.config.scheduler == cli.SCHED_DEOTTE:
            scheduler = get_exp_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=5 * self.steps_per_epoch,
                steps_per_epoch=self.steps_per_epoch,
                num_sustain_steps=0,
            )

        else:
            raise Exception(f"unknown scheduler {self.config.scheduler}")

        return [optimizer], [scheduler]

    # def on_train_start(self):
    #     self.freeze_bn()

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.features.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


def train_cmd(config: cli.RunOptions):

    # monkey patch for multiple test runs
    pl.trainer.distrib_data_parallel.RANDOM_PORTS = pl.trainer.distrib_data_parallel.rng1.randint(
        10000, 19999, 5000
    )

    train_df, train_df_2019, train_df_external, test_df, meta_features = load_dataset(
        config
    )

    if config.dry_run:
        train_df = train_df.head(640)
        train_df_2019 = train_df_2019.head(640)

    tta_transform = get_tta_transforms(config)

    test_transform = A.Compose([A.Normalize(), ToTensorV2()], p=1.0)

    def train_fn(trial):

        seed = 1253
        if config.hpo:
            # config.learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
            # config.weight_decay = trial.suggest_loguniform("wd", 1e-7, 1e-4)
            # print("lr", config.learning_rate, "wd", config.weight_decay)
            config.alpha = trial.suggest_uniform("alpha", 0.1, 0.8)
            config.gamma = trial.suggest_discrete_uniform("gamma", 0.5, 3.0, 0.5)
            print("alpha", config.alpha, "gamma", config.gamma)

        train_transform = get_train_transforms(config)

        seed_everything(seed)

        return train_model_cv(
            train_df=train_df,
            train_df_2018=train_df_2019,
            train_df_external=train_df_external,
            meta_features=meta_features,
            config=config,
            train_transform=train_transform,
            tta_transform=tta_transform,
            test_transform=test_transform,
            trial=trial,
        )

    if config.hpo:
        pruner = (
            optuna.pruners.MedianPruner()
            if config.hpo_pruning
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(
            direction="maximize",
            study_name="isic2020-study",
            storage="sqlite:///hpo_study.db",
            load_if_exists=True,
            pruner=pruner,
        )
        study.optimize(train_fn, n_trials=config.hpo_n_trials)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        fixed_trial = optuna.trial.FixedTrial({"param1": "value1"})
        train_fn(fixed_trial)

    if config.no_cv:
        print(Fore.MAGENTA, "Prediction on --no_cv disabled", Style.RESET_ALL)
        print(Fore.MAGENTA, f"Model saved to {config.output_path}", Style.RESET_ALL)
    else:
        train_transform = get_train_transforms(config)
        predict_model(
            test_df=test_df,
            meta_features=meta_features,
            config=config,
            train_transform=train_transform,
            tta_transform=tta_transform,
            test_transform=test_transform,
        )


class CommanCLI(click.MultiCommand):
    def list_commands(self, ctx):
        rv = ["train"]
        rv.sort()
        return rv

    def get_command(self, ctx, name):
        config = configobj.ConfigObj("config", unrepr=True)
        params = list()
        for opt in cli.options:
            click_opt = click.Option(
                ("--" + opt.name,),
                default=config.get(opt.name, opt.default),
                help=opt.desc,
                is_flag=opt.is_flag,
                type=opt.type,
                callback=opt.callback,
            )
            params.append(click_opt)

        @click.pass_context
        def train_callback(*_args, **kwargs):
            run_options = cli.RunOptions()

            for key, val in kwargs.items():
                run_options.__setattr__(key, val)
            run_options.post_init()
            train_cmd(run_options)

        ret = click.Command(name, params=params, callback=train_callback)
        return ret


@click.command(cls=CommanCLI)
@click.pass_context
def run(_ctx, *_args, **_kwargs):
    pass


if __name__ == "__main__":
    run()
