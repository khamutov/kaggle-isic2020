import os
import random
import time
import datetime

import signal
import sys
from pathlib import Path

import cv2
import pandas as pd
import numpy as np

from colorama import Fore, Style

import click
import click_config_file

from tqdm.auto import tqdm, trange

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchtoolbox.transform as transforms

from torch.multiprocessing import set_start_method

from efficientnet_pytorch import EfficientNet

import warnings


# monkey patch for click START
def get_app_dir(_unused):
    return "."


click.get_app_dir = get_app_dir
# monkey patch END


TEST_PATH = "../kaggle/test.csv"


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
        metadata = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)

        if self.transforms:
            image = self.transforms(image)

        if self.is_train:
            y = self.df.iloc[index]['target']
            #             image = image.cuda()
            return (image, metadata), y
        else:
            return (image, metadata)

    def __len__(self):
        return len(self.df)


class EfficientNetwork(nn.Module):
    def __init__(self, output_size, no_columns, b4=False, b2=False):
        super().__init__()
        self.b4, self.b2, self.no_columns = b4, b2, no_columns

        # Define Feature part (IMAGE)
        if b4:
            self.features = EfficientNet.from_pretrained('efficientnet-b4')
        elif b2:
            self.features = EfficientNet.from_pretrained('efficientnet-b2')
        else:
            self.features = EfficientNet.from_pretrained('efficientnet-b7')

        # (CSV) or Meta Features
        self.csv = nn.Sequential(nn.Linear(self.no_columns, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),

                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),

                                 nn.Linear(250, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3))

        # Define Classification part
        if b4:
            self.classification = nn.Sequential(nn.Linear(1792 + 250, 250),
                                                nn.Linear(250, output_size))
        elif b2:
            self.classification = nn.Sequential(nn.Linear(1408 + 250, 250),
                                                nn.Linear(250, output_size))
        else:
            self.classification = nn.Sequential(nn.Linear(2560 + 250, 250),
                                                nn.Linear(250, output_size))

    def forward(self, image, csv_data, prints=False):

        if prints:
            print('Input Image shape:', image.shape, '\n' +
                  'Input csv_data shape:', csv_data.shape)

        # IMAGE CNN
        image = self.features.extract_features(image)

        if prints:
            print('Features Image shape:', image.shape)

        if self.b4:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1792)
        elif self.b2:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1408)
        else:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 2560)
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


def load_dataset():
    train_df = pd.read_csv('../kaggle-datasource/melanoma-external-malignant-256/train_concat.csv')
    test_df = pd.read_csv(TEST_PATH)

    train_df['sex'] = train_df['sex'].map({'male': 1, 'female': 0})
    test_df['sex'] = test_df['sex'].map({'male': 1, 'female': 0})
    train_df['sex'] = train_df['sex'].fillna(-1)
    test_df['sex'] = test_df['sex'].fillna(-1)

    # imputing
    imp_mean = (train_df["age_approx"].sum()) / (train_df["age_approx"].count() - train_df["age_approx"].isna().sum())
    train_df['age_approx'] = train_df['age_approx'].fillna(imp_mean)
    train_df['age_approx'].head()
    imp_mean_test = (test_df["age_approx"].sum()) / (test_df["age_approx"].count())
    test_df['age_approx'] = test_df['age_approx'].fillna(imp_mean_test)

    train_df['patient_id'] = train_df['patient_id'].fillna(0)

    # OHE
    # TODO: make on sklearn

    concat = pd.concat([train_df['anatom_site_general_challenge'], test_df['anatom_site_general_challenge']],
                       ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]]], axis=1)
    test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]:].reset_index(drop=True)], axis=1)

    meta_features = ['sex', 'age_approx'] + [col for col in train_df.columns if 'site_' in col]
    meta_features.remove('anatom_site_general_challenge')

    test_df = test_df.drop(["anatom_site_general_challenge"], axis=1)
    train_df = train_df.drop(["anatom_site_general_challenge"], axis=1)

    return train_df, test_df, meta_features


def train_model(train_df, meta_features, config):
    output_size = 1  # statics

    train_transform = transforms.Compose([
        #     HairGrowth(hairs = 5,hairs_folder='/kaggle/input/melanoma-hairs/'),
        transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=32. / 255., saturation=0.5, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        #     HairGrowth(hairs = 5,hairs_folder='/kaggle/input/melanoma-hairs/'),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    skf = GroupKFold(n_splits=5)

    train_len = len(train_df)
    oof = np.zeros(shape=(train_len, 1))

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(X=np.zeros(len(train_df)), y=train_df['target'], groups=train_df['patient_id'].tolist()), 1):
        print(Fore.CYAN, '-' * 20, Style.RESET_ALL, Fore.MAGENTA, 'Fold', fold_idx, Style.RESET_ALL, Fore.CYAN,
              '-' * 20,
              Style.RESET_ALL)
        best_val = None
        patience = config[CONFIG_PATIENCE]  # Best validation score within this fold
        model_path = 'model{Fold}.pth'.format(Fold=fold_idx)
        train_dataset = MelanomaDataset(df=train_df.iloc[train_idx].reset_index(drop=True),
                                        imfolder=config[CONFIG_DATASET_MALIGNANT_256] / 'train/train/',
                                        is_train=True,
                                        transforms=train_transform,
                                        meta_features=meta_features)
        val = MelanomaDataset(df=train_df.iloc[val_idx].reset_index(drop=True),
                              imfolder=config[CONFIG_DATASET_MALIGNANT_256] / 'train/train/',
                              is_train=True,
                              transforms=test_transform,
                              meta_features=meta_features)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config[CONFIG_BATCH_SIZE],
                                  shuffle=True,
                                  num_workers=config[CONFIG_NUM_WORKERS],
                                  pin_memory=True)
        val_loader = DataLoader(dataset=val,
                                batch_size=config[CONFIG_BATCH_SIZE],
                                shuffle=False,
                                num_workers=config[CONFIG_NUM_WORKERS],
                                pin_memory=True)

        model = EfficientNetwork(output_size=output_size, no_columns=len(meta_features), b2=True)
        model = model.to(config[CONFIG_DEVICE])

        criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=config[CONFIG_LR], weight_decay=config[CONFIG_DECAY])
        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      mode='max',
                                      patience=config[CONFIG_LR_PATIENCE],
                                      verbose=True,
                                      factor=config[CONFIG_LR_FACTOR])
        for epoch in trange(config[CONFIG_EPOCHES], desc='Epoch'):
            start_time = time.time()
            correct = 0
            train_losses = 0

            model.train()  # Set the model in train mode

            for data, labels in tqdm(train_loader, desc='Batch', leave=False):
                data[0] = torch.tensor(data[0], device=config[CONFIG_DEVICE], dtype=torch.float32)
                data[1] = torch.tensor(data[1], device=config[CONFIG_DEVICE], dtype=torch.float32)
                labels = torch.tensor(labels, device=config[CONFIG_DEVICE], dtype=torch.float32)

                # Clear gradients first; very important, usually done BEFORE prediction
                optimizer.zero_grad()

                # Log Probabilities & Backpropagation
                out = model(data[0], data[1])
                loss = criterion(out, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

                # --- Save information after this batch ---
                # Save loss
                # From log probabilities to actual probabilities
                # 0 and 1
                train_preds = torch.round(torch.sigmoid(out))
                train_losses += loss.item()

                # Number of correct predictions
                correct += (train_preds.cpu() == labels.cpu().unsqueeze(1)).sum().item()

            # Compute Train Accuracy
            train_acc = correct / len(train_idx)
            model.eval()  # switch model to the evaluation mode
            val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=config[CONFIG_DEVICE])
            with torch.no_grad():  # Do not calculate gradient since we are only predicting

                for j, (data_val, label_val) in enumerate(tqdm(val_loader, desc='Val: ', leave=False)):
                    data_val[0] = torch.tensor(data_val[0], device=config[CONFIG_DEVICE], dtype=torch.float32)
                    data_val[1] = torch.tensor(data_val[1], device=config[CONFIG_DEVICE], dtype=torch.float32)
                    label_val = torch.tensor(label_val, device=config[CONFIG_DEVICE], dtype=torch.float32)
                    z_val = model(data_val[0], data_val[1])
                    val_pred = torch.sigmoid(z_val)
                    val_preds[j * data_val[0].shape[0]:j * data_val[0].shape[0] + data_val[0].shape[0]] = val_pred
                val_acc = accuracy_score(train_df.iloc[val_idx]['target'].values, torch.round(val_preds.cpu()))
                val_roc = roc_auc_score(train_df.iloc[val_idx]['target'].values, val_preds.cpu())

                epochval = epoch + 1

                print(Fore.YELLOW, 'Epoch: ', Style.RESET_ALL, epochval, '|', Fore.CYAN, 'Loss: ', Style.RESET_ALL,
                      train_losses, '|', Fore.GREEN, 'Train acc:', Style.RESET_ALL, train_acc, '|', Fore.BLUE,
                      ' Val acc: ', Style.RESET_ALL, val_acc, '|', Fore.RED, ' Val roc_auc:', Style.RESET_ALL, val_roc,
                      '|', Fore.YELLOW, ' Training time:', Style.RESET_ALL,
                      str(datetime.timedelta(seconds=time.time() - start_time)))

                scheduler.step(val_roc)
                # During the first iteration (first epoch) best validation is set to None
                if not best_val:
                    best_val = val_roc  # So any validation roc_auc we have is the best one for now
                    torch.save(model, model_path)  # Saving the model
                    continue

                if val_roc >= best_val:
                    best_val = val_roc
                    patience = patience  # Resetting patience since we have new best validation accuracy
                    torch.save(model, model_path)  # Saving current best model
                else:
                    patience -= 1
                    if patience == 0:
                        print(Fore.BLUE, 'Early stopping. Best Val roc_auc: {:.3f}'.format(best_val), Style.RESET_ALL)
                        break

        model = torch.load(model_path)  # Loading best model of this fold
        model.eval()  # switch model to the evaluation mode
        val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=config[CONFIG_DEVICE])
        with torch.no_grad():
            # Predicting on validation set once again to obtain data for OOF
            for j, (x_val, y_val) in enumerate(val_loader):
                x_val[0] = torch.tensor(x_val[0], device=config[CONFIG_DEVICE], dtype=torch.float32)
                x_val[1] = torch.tensor(x_val[1], device=config[CONFIG_DEVICE], dtype=torch.float32)
                y_val = torch.tensor(y_val, device=config[CONFIG_DEVICE], dtype=torch.float32)
                z_val = model(x_val[0], x_val[1])
                val_pred = torch.sigmoid(z_val)
                val_preds[j * x_val[0].shape[0]:j * x_val[0].shape[0] + x_val[0].shape[0]] = val_pred
            oof[val_idx] = val_preds.cpu().numpy()


@click.group()
def cli():
    pass


CONFIG_EPOCHES = "epochs"
CONFIG_BATCH_SIZE = "batch_size"
CONFIG_PATIENCE = "patience"
CONFIG_LR_PATIENCE = "lr_patience"
CONFIG_LR = "learning_rate"
CONFIG_DECAY = "weight_decay"
CONFIG_LR_FACTOR = "lr_factor"
CONFIG_NUM_WORKERS = "num_workers"
CONFIG_DATASET_MALIGNANT_256 = "dataset_malignant_256"
CONFIG_DEVICE = "device"


@cli.command()
@click.option('--' + CONFIG_EPOCHES, default=10, help='Number of epoches for training.')
@click.option('--' + CONFIG_BATCH_SIZE, default=64, help='Train batch size.')
@click.option('--' + CONFIG_PATIENCE, default=3, help='early stopping patience')
@click.option('--' + CONFIG_LR_PATIENCE, default=1, help='patience for learning rate')
@click.option('--' + CONFIG_LR, default=0.001, help='Learning Rate')
@click.option('--' + CONFIG_DECAY, default=0.0, help='Decay Factor')
@click.option('--' + CONFIG_LR_FACTOR, default=0.4, help='')
@click.option('--' + CONFIG_NUM_WORKERS, default=0, help='number of subprocess to use while data loading')
@click.option('--' + CONFIG_DATASET_MALIGNANT_256, help='path to external malignant-256 dataset',
              type=click.Path(exists=True))
@click.option('--' + CONFIG_DEVICE, help='device: cpu, cuda, cuda:1')
@click_config_file.configuration_option(implicit=True, config_file_name="config")
def train(**kwargs):
    config = kwargs

    config[CONFIG_DATASET_MALIGNANT_256] = Path(config[CONFIG_DATASET_MALIGNANT_256])

    tqdm.pandas()
    warnings.filterwarnings("ignore")

    seed = 1234
    seed_everything(seed)

    if not config.get(CONFIG_DEVICE):
        config[CONFIG_DEVICE] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # check device available
        tmp_tensor = torch.rand(1).to(config['device'])
        del tmp_tensor

    train_df, test_df, meta_features = load_dataset()

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    train_model(train_df=train_df, meta_features=meta_features, config=config)


if __name__ == '__main__':
    cli()
