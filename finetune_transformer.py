from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import TransformerRegressor, Transformer, regressoionHead
from model.utils import *
from datetime import datetime, timedelta
from time import time
from torch.utils.data import dataset, DataLoader

import os
import csv
import yaml
import shutil
import argparse
import sys
import time
import warnings
import numpy as np
import pandas as pd
from random import sample
from sklearn import metrics
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_finetune_transformer import MOF_ID_Dataset

warnings.simplefilter("ignore")


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_ft_transformer.yaml', os.path.join(model_checkpoints_folder, 'config_ft_transformer.yaml'))


class FineTune(object):
    def __init__(self, config, log_dir):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(log_dir=log_dir)

        self.random_seed = self.config['dataloader']['randomSeed']

        # Load dataset
        self.data = pd.read_excel(self.config['dataset']['dataPath'])
        self.feature_columns = self.config['dataset']['feature_columns']
        self.target_column = self.config['dataset']['target_column']

        # Tokenizer
        self.vocab_path = self.config['vocab_path']
        self.tokenizer = MOFTokenizer(self.vocab_path, model_max_length=512, padding_side='right')

        # Split data
        valid_ratio = self.config['dataloader']['valid_ratio']
        test_ratio = self.config['dataloader']['test_ratio']

        self.train_data, self.valid_data, self.test_data = split_data(
            self.data, valid_ratio=valid_ratio, test_ratio=test_ratio, randomSeed=self.random_seed
        )

        # Create datasets and loaders
        self.train_dataset = MOF_ID_Dataset(data=self.train_data, tokenizer=self.tokenizer)
        self.valid_dataset = MOF_ID_Dataset(data=self.valid_data, tokenizer=self.tokenizer)
        self.test_dataset = MOF_ID_Dataset(data=self.test_data, tokenizer=self.tokenizer)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'],
            shuffle=True, drop_last=False, pin_memory=False
        )
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'],
            shuffle=False, drop_last=False, pin_memory=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'],
            shuffle=False, drop_last=False, pin_memory=False
        )

        self.criterion = nn.MSELoss()
        self.normalizer = Normalizer(torch.from_numpy(self.train_dataset.label))

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        print("Running on:", device)
        return device

    # Remaining methods unchanged (train, _load_pre_trained_weights, _validate, test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer finetuning')
    parser.add_argument('--seed', default=1, type=int, metavar='Seed', help='random seed for splitting data (default: 1)')

    args = parser.parse_args(sys.argv[1:])
    config = yaml.load(open("config_ft_transformer.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    config['dataloader']['randomSeed'] = args.seed

    ptw = config['trained_with']
    seed = config['dataloader']['randomSeed']

    task_name = "YOUR_TASK_NAME"  # Replace this with your task name
    log_dir = os.path.join(
        'training_results/finetuning/Transformer',
        f'Trans_{config["dataset"]["data_name"]}_{args.seed}'
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fine_tune = FineTune(config, log_dir)
    fine_tune.train()
    loss, metric = fine_tune.test()

    fn = f'Trans_{ptw}_{task_name}_{seed}.csv'
    print(fn)
    df = pd.DataFrame([[loss, metric.item()]])
    df.to_csv(os.path.join(log_dir, fn), mode='a', index=False, header=False)
