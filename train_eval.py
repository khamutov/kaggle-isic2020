import datetime
import math
import os
import random
import signal
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import click
import configobj
import cv2
import mlflow
import numpy as np
import pandas as pd
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtoolbox.transform as transforms
from colorama import Fore, Style
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from torch.multiprocessing import set_start_method
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm, trange

from augmentation.hairs import AdvancedHairAugmentation

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2

import cli

FOLDS = 5


def get_train_transforms(config):
    return A.Compose([
            AdvancedHairAugmentation(hairs_folder='/home/a.khamutov/kaggle-datasource/melanoma-hairs')
            if config.advanced_hair_augmentation else A.NoOp(),
            A.JpegCompression(p=0.5) if config.jpeg_compression else A.NoOp(),
            A.Rotate(limit=80, p=1.0) if config.rotate else A.NoOp(),
            A.OneOf([
                A.OpticalDistortion() if config.optical_distortion else A.NoOp(),
                A.GridDistortion() if config.grid_distortion else A.NoOp(),
                A.IAAPiecewiseAffine() if config.piecewise_affine else A.NoOp(),
            ]),
            # A.RandomSizedCrop(min_max_height=(int(resolution*0.7), input_res),
            #                   height=resolution, width=resolution, p=1.0),
            A.HorizontalFlip(p=0.5) if not config.horizontal_flip else A.NoOp(),
            A.VerticalFlip(p=0.5) if not config.vertical_flip else A.NoOp(),
            A.GaussianBlur(p=0.3) if not config.gaussian_blur else A.NoOp(),
            A.OneOf([
                A.RandomBrightnessContrast() if not config.random_brightness_contrast else A.NoOp(),
                A.HueSaturationValue(hue_shift_limit=0) if not config.hue_saturation_value else A.NoOp(),
            ]),
            A.Cutout(num_holes=8,
                     max_h_size=config.input_size//8,
                     max_w_size=config.input_size//8,
                     fill_value=0,
                     p=0.3) if not config.cutout else A.NoOp(),
            A.Normalize(),
            ToTensorV2(),
        ], p=1.0)


def signal_handler(_sig, _frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(seed_value)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(seed_value)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = True


# from huggingface transformers
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
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
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# This is a common train schedule for transfer learning.
# The learning rate starts near zero, then increases to a maximum, then decays over time.
def get_exp_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, steps_per_epoch: int, num_sustain_steps: int = 0, lr_decay: float = 0.8):
    def lrfn(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + num_sustain_steps:
            return 1.0

        return lr_decay ** (float(current_step - num_warmup_steps - num_sustain_steps) / steps_per_epoch)

    return LambdaLR(optimizer, lrfn)


class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, is_train: bool = True, transforms=None, meta_features=None):

        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.is_train = is_train
        self.meta_features = meta_features

    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        metadata = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)

        if self.transforms:
            sample = self.transforms(image=image)
            image = sample['image']

        if self.is_train:
            y = self.df.iloc[index]['target']
            #             image = image.cuda()
            return (image, metadata), y
        else:
            return (image, metadata)

    def __len__(self):
        return len(self.df)


class EfficientNetwork(nn.Module):
    def __init__(self, output_size, no_columns, model_name='efficientnet-b0'):
        super().__init__()
        self.no_columns = no_columns

        self.features = EfficientNet.from_pretrained(model_name)

        # (CSV) or Meta Features
        meta_features_out = 250
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 250),
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
                                 nn.Dropout(p=0.3))

        self.eff_net_out_features = getattr(self.features, '_fc').in_features

        fc_hidden_size = 250
        self.classification = nn.Sequential(nn.Linear(self.eff_net_out_features + meta_features_out, fc_hidden_size),
                                            nn.Linear(fc_hidden_size, output_size))

    def forward(self, image, csv_data, prints=False):

        if prints:
            print('Input Image shape:', image.shape, '\n' +
                  'Input csv_data shape:', csv_data.shape)

        # IMAGE CNN
        image = self.features.extract_features(image)

        if prints:
            print('Features Image shape:', image.shape)

        # image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, self.eff_net_out_features)
        features = F.adaptive_avg_pool2d(image, 1)
        image = features.view(features.size(0), -1)
        if prints:
            print('Image Reshaped shape:', image.shape)

        # CSV FNN
        csv_data = self.csv(csv_data)
        if prints:
            print('CSV Data:', csv_data.shape)

        # Concatenate
        image_csv_data = torch.cat((image, csv_data), dim=1)

        # CLASSIF
        out = self.classification(image_csv_data)
        if prints:
            print('Out shape:', out.shape)

        return out


def load_dataset(config: cli.RunOptions):
    train_df = pd.read_csv(config.dataset_2020() / 'train.csv')
    train_df_2019 = pd.read_csv(config.dataset_2019() / 'train.csv')

    train_df['fold'] = train_df['tfrecord']
    train_df_2019['fold'] = train_df_2019['tfrecord']
    test_df = pd.read_csv(config.dataset_official / 'test.csv')

    train_df['sex'] = train_df['sex'].map({'male': 1, 'female': 0})
    train_df_2019['sex'] = train_df_2019['sex'].map({'male': 1, 'female': 0})

    test_df['sex'] = test_df['sex'].map({'male': 1, 'female': 0})

    train_df['sex'] = train_df['sex'].fillna(-1)
    train_df_2019['sex'] = train_df_2019['sex'].fillna(-1)

    test_df['sex'] = test_df['sex'].fillna(-1)

    # imputing
    imp_mean = (train_df["age_approx"].sum()) / (train_df["age_approx"].count() - train_df["age_approx"].isna().sum())
    train_df['age_approx'] = train_df['age_approx'].fillna(imp_mean)
    train_df_2019['age_approx'] = train_df_2019['age_approx'].fillna(imp_mean)

    imp_mean_test = (test_df["age_approx"].sum()) / (test_df["age_approx"].count())
    test_df['age_approx'] = test_df['age_approx'].fillna(imp_mean_test)

    train_df['patient_id'] = train_df['patient_id'].fillna(0)
    train_df_2019['patient_id'] = train_df_2019['patient_id'].fillna(0)

    # OHE
    # TODO: make on sklearn

    concat = pd.concat([train_df['anatom_site_general_challenge'],
                        train_df_2019['anatom_site_general_challenge'],
                        test_df['anatom_site_general_challenge']],
                       ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]]], axis=1)
    train_df_2019 = pd.concat([train_df_2019, dummies.iloc[train_df.shape[0]:train_df.shape[0]+train_df_2019.shape[0]].reset_index(drop=True)], axis=1)
    test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]+train_df_2019.shape[0]:].reset_index(drop=True)], axis=1)

    meta_features = ['sex', 'age_approx'] + [col for col in train_df.columns if 'site_' in col]
    meta_features.remove('anatom_site_general_challenge')

    test_df = test_df.drop(["anatom_site_general_challenge"], axis=1)
    train_df = train_df.drop(["anatom_site_general_challenge"], axis=1)
    train_df_2019 = train_df_2019.drop(["anatom_site_general_challenge"], axis=1)

    return train_df, train_df_2019, test_df, meta_features


@dataclass
class TrainResult:
    best_val: float
    pred = None
    pred_tta = None
    target = None
    names = None

    def __init__(self):
        pass


def train_fit(train_df, train_df_2018, val_df, train_transform, test_transform, meta_features, config: cli.RunOptions, fold_idx=0) -> TrainResult:
    output_size = 1  # statics

    train_result = TrainResult()
    train_result.names = val_df["image_name"].to_numpy()

    best_val = None
    patience = config.patience  # Best validation score within this fold

    model_path = Path(config.output_path) / f"model{fold_idx}.pth"

    train_dataset_2020 = MelanomaDataset(df=train_df,
                                         imfolder=config.dataset_2020() / 'train',
                                         is_train=True,
                                         transforms=train_transform,
                                         meta_features=meta_features)
    train_dataset_2018 = MelanomaDataset(df=train_df_2018,
                                         imfolder=config.dataset_2019() / 'train',
                                         is_train=True,
                                         transforms=train_transform,
                                         meta_features=meta_features)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_2020, train_dataset_2018])
    val = MelanomaDataset(df=val_df,
                          imfolder=config.dataset_2020() / 'train',
                          is_train=True,
                          transforms=test_transform,
                          meta_features=meta_features)
    val_tta = MelanomaDataset(df=val_df,
                              imfolder=config.dataset_2020() / 'train',
                              is_train=True,
                              transforms=train_transform,
                              meta_features=meta_features)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(dataset=val,
                            batch_size=config.batch_size * 2,
                            shuffle=False,
                            num_workers=config.num_workers,
                            pin_memory=True)
    val_tta_loader = DataLoader(dataset=val_tta,
                                batch_size=config.batch_size * 2,
                                shuffle=False,
                                num_workers=config.num_workers,
                                pin_memory=True)

    model = EfficientNetwork(output_size=output_size, no_columns=len(meta_features), model_name=config.model)
    model = model.to(config.device)

    pos_weight = torch.tensor([10]).to(config.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if config.optim == cli.OPTIM_ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optim == cli.OPTIM_ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optim == cli.OPTIM_SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, nesterov=True)
    else:
        raise Exception(f"unknown optimizer f{config.optim}")

    # scheduler = ReduceLROnPlateau(optimizer=optimizer,
    #                               mode='max',
    #                               patience=config.patience,
    #                               verbose=True,
    #                               factor=config.lr_factor)
    steps_per_epoch = int(len(train_dataset) / config.batch_size)

    if config.scheduler == cli.SCHED_1CYC:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            max_lr=config.learning_rate,
            epochs=config.epochs,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
            base_momentum=0.90,
            max_momentum=0.95,
        )
    elif config.scheduler == cli.SCHED_COSINE:
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=3 * steps_per_epoch,
                                                    num_training_steps=config.epochs * steps_per_epoch)
    elif config.scheduler == cli.SCHED_DEOTTE:
        scheduler = get_exp_schedule_with_warmup(optimizer=optimizer,
                                                 num_warmup_steps=5 * steps_per_epoch,
                                                 steps_per_epoch=steps_per_epoch,
                                                 num_sustain_steps=0)

    else:
        raise Exception(f"unknown scheduler {config.scheduler}")

    for epoch in trange(config.epochs, desc='Epoch'):
        start_time = time.time()
        correct = 0
        train_losses = 0
        val_losses = 0

        model.train()  # Set the model in train mode

        for data, labels in tqdm(train_loader, desc='Batch', leave=False):
            data[0] = torch.tensor(data[0], device=config.device, dtype=torch.float32)
            data[1] = torch.tensor(data[1], device=config.device, dtype=torch.float32)
            labels = torch.tensor(labels, device=config.device, dtype=torch.float32)

            y_smooth = labels.float() * (1 - config.loss_bce_label_smoothing) + 0.5 * config.loss_bce_label_smoothing

            # Clear gradients first; very important, usually done BEFORE prediction
            optimizer.zero_grad()

            # Log Probabilities & Backpropagation
            out = model(data[0], data[1])
            loss = criterion(out, y_smooth.unsqueeze(1))
            loss.backward()
            optimizer.step()

            scheduler.step()

            # --- Save information after this batch ---
            # Save loss
            # From log probabilities to actual probabilities
            # 0 and 1
            train_preds = torch.round(torch.sigmoid(out))
            train_losses += loss.item()

            # Number of correct predictions
            correct += (train_preds.cpu() == labels.cpu().unsqueeze(1)).sum().item()

        # Compute Train Accuracy
        train_acc = correct / len(train_dataset)
        model.eval()  # switch model to the evaluation mode
        val_pred_arr = []
        with torch.no_grad():  # Do not calculate gradient since we are only predicting

            for j, (data_val, label_val) in enumerate(tqdm(val_loader, desc='Val: ', leave=False)):
                data_val[0] = torch.tensor(data_val[0], device=config.device, dtype=torch.float32)
                data_val[1] = torch.tensor(data_val[1], device=config.device, dtype=torch.float32)
                label_val = torch.tensor(label_val, device=config.device, dtype=torch.float32)

                y_smooth = label_val.float() * (1 - config.loss_bce_label_smoothing) + 0.5 * config.loss_bce_label_smoothing

                z_val = model(data_val[0], data_val[1])

                loss = criterion(z_val, y_smooth.unsqueeze(1))
                val_losses += loss.item()

                val_pred = torch.sigmoid(z_val)
                val_pred_arr.append(val_pred.cpu().numpy())
            val_preds = np.concatenate(val_pred_arr)
            val_acc = accuracy_score(val_df['target'].values, np.round(val_preds))
            val_roc = roc_auc_score(val_df['target'].values, val_preds)

            epochval = epoch + 1

            train_loss = train_losses / len(train_dataset)
            val_loss = val_losses / len(val)

            print(Fore.YELLOW, 'Epoch: ', Style.RESET_ALL, epochval, '|',
                  Fore.CYAN, 'Loss: ', Style.RESET_ALL, train_loss, '|',
                  Fore.BLUE, ' Val loss: ', Style.RESET_ALL, val_loss, '|',
                  Fore.RED, ' Val roc_auc:', Style.RESET_ALL, val_roc, '|',
                  Fore.YELLOW, ' Training time:', Style.RESET_ALL,
                  str(datetime.timedelta(seconds=time.time() - start_time)))

            if config.is_track_mlflow():
                mlflow.active_run()
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                mlflow.log_metric("val_roc_auc", val_roc, step=epoch)

            # scheduler.step(val_roc)
            # During the first iteratsion (first epoch) best validation is set to None
            if not best_val:
                best_val = val_roc  # So any validation roc_auc we have is the best one for now
                torch.save(model, model_path)  # Saving the model
                continue

            if val_roc >= best_val:
                best_val = val_roc
                patience = config.patience  # Resetting patience since we have new best validation accuracy
                torch.save(model, model_path)  # Saving current best model
            else:
                patience -= 1
                if patience == 0:
                    print(Fore.BLUE, 'Early stopping. Best Val roc_auc: {:.3f}'.format(best_val), Style.RESET_ALL)
                    break

    # val on best model
    model = torch.load(model_path)  # Loading best model of this fold
    model.eval()  # switch model to the evaluation mode
    with torch.no_grad():
        # Predicting on validation set once again to obtain data for OOF
        fold_preds = torch.zeros((len(val_tta), 1), dtype=torch.float32, device=config.device)
        fold_preds_tta = torch.zeros((len(val_tta), 1), dtype=torch.float32, device=config.device)
        val_targets_arr = []

        for i, (x_val, y_val) in enumerate(tqdm(val_loader, desc='Predict no TTA', leave=False)):
            x_val[0] = torch.tensor(x_val[0], device=config.device, dtype=torch.float32)
            x_val[1] = torch.tensor(x_val[1], device=config.device, dtype=torch.float32)
            y_val = torch.tensor(y_val, device=config.device, dtype=torch.float32)
            z_val = model(x_val[0], x_val[1])
            val_pred = torch.sigmoid(z_val)

            fold_preds[i * val_loader.batch_size:i * val_loader.batch_size + x_val[0].shape[0]] += val_pred
            val_targets_arr.append(y_val.cpu().numpy())

        for _ in trange(config.tta, desc='TTA', leave=False):
            for i, (x_val, y_val) in enumerate(tqdm(val_tta_loader, desc='Predict TTA', leave=False)):
                x_val[0] = torch.tensor(x_val[0], device=config.device, dtype=torch.float32)
                x_val[1] = torch.tensor(x_val[1], device=config.device, dtype=torch.float32)

                z_val = model(x_val[0], x_val[1])
                val_pred = torch.sigmoid(z_val)

                fold_preds_tta[i * val_tta_loader.batch_size:i * val_tta_loader.batch_size + x_val[0].shape[0]] += val_pred
        fold_preds_tta /= config.tta

        train_result.pred = fold_preds.cpu().numpy()[:, 0]
        train_result.pred_tta = fold_preds_tta.cpu().numpy()[:, 0]
        train_result.target = np.concatenate(val_targets_arr)

    train_result.best_val = best_val

    return train_result


def train_model_no_cv(train_df, train_df_2018, meta_features, config, train_transform, test_transform):
    train_len = len(train_df)
    oof = np.zeros(shape=(train_len, 1))
    oof_pred = []
    oof_pred_tta = []
    oof_target = []
    oof_val = []
    oof_folds = []
    oof_names = []

    idxT = np.arange(12)
    idxV = np.arange(12, 15)
    fold_idx = 1

    train_idx = train_df.loc[train_df['fold'].isin(idxT)].index
    train_idx_2018 = train_df_2018.loc[train_df_2018['fold'].isin(idxT*2)].index
    val_idx = train_df.loc[train_df['fold'].isin(idxV)].index

    oof_names.append(train_df.iloc[val_idx]["image_name"].to_numpy())

    run_info = None
    if config.is_track_mlflow():
        run_info = mlflow.start_run(nested=True, run_name="Fold {}".format(fold_idx))
        mlflow.log_param("fold", fold_idx)

    print(Fore.CYAN, '-' * 20, Style.RESET_ALL, Fore.MAGENTA, 'No CV mode', fold_idx, Style.RESET_ALL, Fore.CYAN,
          '-' * 20,
          Style.RESET_ALL)

    train_fit_df = train_df.iloc[train_idx].reset_index(drop=True)
    train_fit_df_2018 = train_df_2018.iloc[train_idx_2018].reset_index(drop=True)
    val_fit_df = train_df.iloc[val_idx].reset_index(drop=True)

    train_result = train_fit(train_df=train_fit_df,
                             train_df_2018=train_fit_df_2018,
                             val_df=val_fit_df,
                             train_transform=train_transform,
                             test_transform=test_transform,
                             meta_features=meta_features,
                             config=config,
                             mlflow_run_info=run_info,
                             fold_idx=fold_idx)

    oof_pred.append(train_result.pred)
    oof_pred_tta.append(train_result.pred_tta)
    oof_target.append(train_result.target)
    oof_folds.append(np.ones_like(oof_target[-1], dtype='int8') * fold_idx)

    if config.is_track_mlflow():
        mlflow.log_metric("best_roc_auc", train_result.best_val)
        mlflow.end_run()

    oof = np.concatenate(oof_pred).squeeze()
    oof_tta = np.concatenate(oof_pred_tta).squeeze()
    true = np.concatenate(oof_target)
    folds = np.concatenate(oof_folds)
    auc = roc_auc_score(true, oof)
    auc_tta = roc_auc_score(true, oof_tta)
    names = np.concatenate(oof_names)

    print(Fore.CYAN, '-' * 60, Style.RESET_ALL)
    print(Fore.MAGENTA, 'OOF ROC AUC    ', auc, Style.RESET_ALL)
    print(Fore.MAGENTA, 'OOF ROC AUC TTA', auc_tta, Style.RESET_ALL)
    print(Fore.CYAN, '-' * 60, Style.RESET_ALL)

    if config.is_track_mlflow():
        mlflow.log_metric("oof_roc_auc", auc)
        mlflow.log_metric("oof_roc_auc_tta", auc_tta)

    # SAVE OOF TO DISK
    df_oof = pd.DataFrame(dict(image_name=names, target=true, pred=oof, fold=folds))
    df_oof.to_csv('oof.csv', index=False)

def train_model_cv(train_df, train_df_2018, meta_features, config, train_transform, test_transform) -> str:
    train_len = len(train_df)
    oof = np.zeros(shape=(train_len, 1))
    oof_pred = []
    oof_target = []
    oof_val = []
    oof_folds = []
    oof_names = []

    model_path = ""
    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=47)
    for fold_idx, (idxT, idxV) in enumerate(skf.split(np.arange(15)), 1):
        train_idx = train_df.loc[train_df['fold'].isin(idxT)].index
        train_idx_2018 = train_df_2018.loc[train_df_2018['fold'].isin(idxT * 2)].index
        val_idx = train_df.loc[train_df['fold'].isin(idxV)].index

        oof_names.append(train_df.iloc[val_idx]["image_name"].to_numpy())

        run_info = None
        if config.is_track_mlflow():
            run_info = mlflow.start_run(nested=True, run_name="Fold {}".format(fold_idx))
            mlflow.log_param("fold", fold_idx)

        print(Fore.CYAN, '-' * 20, Style.RESET_ALL, Fore.MAGENTA, 'Fold', fold_idx, Style.RESET_ALL, Fore.CYAN,
              '-' * 20,
              Style.RESET_ALL)

        train_fit_df = train_df.iloc[train_idx].reset_index(drop=True)
        train_fit_df_2018 = train_df_2018.iloc[train_idx_2018].reset_index(drop=True)
        val_fit_df = train_df.iloc[val_idx].reset_index(drop=True)

        train_result, model_path = train_fit(train_df=train_fit_df,
                                 train_df_2018=train_fit_df_2018,
                                 val_df=val_fit_df,
                                 train_transform=train_transform,
                                 test_transform=test_transform,
                                 meta_features=meta_features,
                                 config=config,
                                 mlflow_run_info=run_info,
                                 fold_idx=fold_idx)

        oof_pred.append(train_result.pred)
        oof_target.append(train_result.target)
        oof_folds.append(np.ones_like(oof_target[-1], dtype='int8') * fold_idx)

        if config.is_track_mlflow():
            mlflow.log_metric("best_roc_auc", train_result.best_val)
            mlflow.end_run()

    oof = np.concatenate(oof_pred).squeeze()
    true = np.concatenate(oof_target)
    folds = np.concatenate(oof_folds)
    auc = roc_auc_score(true, oof)
    names = np.concatenate(oof_names)

    print(Fore.CYAN, '-' * 60, Style.RESET_ALL)
    print(Fore.MAGENTA, 'OOF ROC AUC', auc, Style.RESET_ALL)
    print(Fore.CYAN, '-' * 60, Style.RESET_ALL)

    if config.is_track_mlflow():
        mlflow.log_metric("oof_roc_auc", auc)

    # SAVE OOF TO DISK
    df_oof = pd.DataFrame(dict(image_name=names, target=true, pred=oof, fold=folds))
    df_oof.to_csv('oof.csv', index=False)
    return model_path


def predict_model(test_df, meta_features, config: cli.RunOptions, train_transform, test_transform):
    print(Fore.MAGENTA, 'Run prediction', Style.RESET_ALL)

    test = MelanomaDataset(df=test_df,
                           imfolder=config.dataset_2020() / 'test',
                           is_train=False,
                           transforms=train_transform,
                           meta_features=meta_features)
    test_loader = DataLoader(dataset=test,
                             batch_size=config.batch_size * 2,
                             shuffle=False,
                             num_workers=config.num_workers,
                             pin_memory=True)

    preds = torch.zeros((len(test), 1), dtype=torch.float32, device=config.device)
    for fold_idx in trange(1, FOLDS + 1, desc="Fold"):
        model = torch.load(Path(config.output_path) / f"model{fold_idx}.pth")
        model.eval()  # switch model to the evaluation mode

        fold_preds = torch.zeros((len(test), 1), dtype=torch.float32, device=config.device)
        with torch.no_grad():
            for _ in trange(config.tta, desc='TTA', leave=False):
                for i, x_test in enumerate(tqdm(test_loader, desc='Predict', leave=False)):
                    x_test[0] = torch.tensor(x_test[0], device=config.device, dtype=torch.float32)
                    x_test[1] = torch.tensor(x_test[1], device=config.device, dtype=torch.float32)
                    z_test = model(x_test[0], x_test[1])
                    z_test = torch.sigmoid(z_test)
                    fold_preds[i * test_loader.batch_size:i * test_loader.batch_size + x_test[0].shape[0]] += z_test
            fold_preds /= config.tta
        preds += fold_preds

    preds /= FOLDS

    submission = pd.DataFrame(dict(image_name=test_df['image_name'].to_numpy(), target=preds.cpu().numpy()[:, 0]))
    submission = submission.sort_values('image_name')
    submission.to_csv('submission.csv', index=False)

    print(Fore.MAGENTA, 'saved to submission.csv', Style.RESET_ALL)


def train_cmd(config: cli.RunOptions):


    train_df, train_df_2019, test_df, meta_features = load_dataset(config)

    if config.dry_run:
        train_df = train_df.head(640)
        train_df_2019 = train_df_2019.head(640)

    train_transform = get_train_transforms(config)
    # train_transform = transforms.Compose([
    #     AdvancedHairAugmentation(hairs_folder='/home/a.khamutov/kaggle-datasource/melanoma-hairs')
    #         if config.hair_augment
    #         else identity,
    #     transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
    #     transforms.RandomApply([
    #         transforms.RandomChoice([
    #                                     transforms.RandomAffine(degrees=20),
    #                                     transforms.RandomAffine(degrees=0, scale=(0.1, 0.15)),
    #                                     transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    #                                     # transforms.RandomAffine(degrees=0,shear=0.15),
    #                                     transforms.RandomHorizontalFlip(p=1.0)
    #                                 ])
    #     ], p=0.5),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ColorJitter(brightness=32. / 255., contrast=0.2, saturation=0.3, hue=0.01),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # test_transform = transforms.Compose([
    #     AdvancedHairAugmentation(hairs_folder='/home/a.khamutov/kaggle-datasource/melanoma-hairs')
    #         if config.hair_augment
    #         else identity,
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    test_transform = A.Compose([
        AdvancedHairAugmentation(hairs_folder='/home/a.khamutov/kaggle-datasource/melanoma-hairs')
        if config.advanced_hair_augmentation else A.NoOp(),
        A.Normalize(),
        ToTensorV2(),
    ], p=1.0)

    train_fn = train_model_no_cv if config.no_cv else train_model_cv
    train_fn(train_df=train_df,
             train_df_2018=train_df_2019,
             meta_features=meta_features,
             config=config,
             train_transform=train_transform,
             test_transform=test_transform)

    if config.no_cv:
        print(Fore.MAGENTA, 'Prediction on --no_cv disabled', Style.RESET_ALL)
    else:
        predict_model(test_df=test_df,
                      meta_features=meta_features,
                      config=config,
                      train_transform=train_transform,
                      test_transform=test_transform)


class CommanCLI(click.MultiCommand):

    def list_commands(self, ctx):
        rv = ['train']
        rv.sort()
        return rv

    def get_command(self, ctx, name):
        config = configobj.ConfigObj("config", unrepr=True)
        params = list()
        for opt in cli.options:
            click_opt = click.Option(("--" + opt.name,),
                                     default=config.get(opt.name, opt.default),
                                     help=opt.desc,
                                     is_flag=opt.is_flag,
                                     type=opt.type,
                                     callback=opt.callback)
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


if __name__ == '__main__':
    run()
