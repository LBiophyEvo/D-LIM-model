"""
alm.py
-----------------------

This script trains an Additive Latent Model (ALM), which learns an additive
latent representation of mutations or features and uses a neural network to 
map that latent space to a fitness (or response) score.

The ALM is similar to the LANTERN model, but here the mapping from latent 
space to fitness is parameterized by a small neural network. It supports 
different datasets such as "harry", "elife", and "protein_inter".

The script:
    1. Loads dataset and defines model hyperparameters.
    2. Trains the model over multiple random subsamples of the data.
    3. Evaluates performance using Pearson correlation on a held-out set.
    4. Saves results (correlations per training fraction) as a `.joblib` file.

Usage:
    python alm.py --flag epistasis --data_flag harry

Arguments:
    --flag : str
        Type of data split or experimental condition ('fitness', 'epistasis', etc.)
    --data_flag : str
        Dataset name ('harry', 'elife', 'protein_inter')

Outputs:
    - Trained model results saved to:
      results/<data_flag>_<flag>/reg_ALM.joblib
"""

# ===== Imports =====
import sys
sys.path.append('../../')  # Ensure project modules can be imported

import os
import joblib
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from scipy.stats import pearsonr
from numpy import mean, std, logspace
from numpy.random import choice, seed
from sklearn.metrics import r2_score

# Local modules
from src.model import Regression, Add_Latent
from src.utils import Data_model, train, train_reg


# ===== Argument Parsing =====
parser = argparse.ArgumentParser(description='Train Additive Latent Model (ALM)')
parser.add_argument('--flag', type=str, default='epistasis',
                    help="Task or condition (e.g., 'fitness', 'epistasis')")
parser.add_argument('--data_flag', type=str, default='harry',
                    help="Dataset to use ('harry', 'elife', 'protein_inter')")
config = parser.parse_args()

flag = config.flag
model_name = 'ALM'


# ===== Define Hyperparameters Based on Dataset =====
if config.data_flag == 'harry':
    hparam = {
        'nb_state': 50,     # Number of latent embedding states
        'emb': 2,           # Latent dimension (2D)
        'hid': 8,           # Hidden layer units
        'nb_layer': 1,      # Number of hidden layers in neural network
        'lr': 1e-2,         # Learning rate
        'wei_dec': 1e-3,    # Weight decay for regularization
        'nb_epoch': 300,    # Number of training epochs
        'bsize': 64,        # Batch size
        'sep': ','          # CSV delimiter
    }
    # Dataset path selection based on task
    if flag == 'fitness':
        data_path = "../data/data_env_1.csv"
    elif flag == 'epistasis':
        data_path = "../data/data_epis_1.csv"

elif config.data_flag == 'elife':
    hparam = {
        'nb_state': 50,
        'emb': 6,
        'hid': 128,
        'nb_layer': 2,
        'lr': 1e-3,
        'wei_dec': 1e-3,
        'nb_epoch': 300,
        'bsize': 128,
        'sep': '@'
    }
    if flag == 'subtle':
        data_path = "../data/elife/elife_data_subtle_env.csv"
    else:
        data_path = "../data/elife/elife_data_strong_env.csv"

elif config.data_flag == 'protein_inter':
    hparam = {
        'nb_state': 50,
        'emb': 6,
        'hid': 128,
        'nb_layer': 2,
        'lr': 1e-3,
        'wei_dec': 1e-3,
        'nb_epoch': 300,
        'bsize': 256,
        'sep': ','
    }
    data_path = "../data/protein_inter/tables1.csv"

else:
    raise ValueError(f"Unknown data_flag: {config.data_flag}")


# ===== Data Loading =====
seed(42)  # Ensure reproducibility
data = Data_model(data_path, 2, sep=hparam['sep'])

# Define validation split fraction
val_frac = logspace(-1, 0.1, num=7)  # Varying fractions of training data (10% to ~1.26×)

# Split into train and validation sets
val_id = choice(range(data.data.shape[0]), int(data.data.shape[0] * 0.3))
train_full_id = [i for i in range(data.data.shape[0]) if i not in val_id]
train_full_data = data[train_full_id, :]
val_data = data[val_id, :]


# ===== Training Function =====
def run_one(i, frac):
    """
    Train the ALM on a random subset of training data and evaluate on validation data.

    Args:
        i (int): Random seed index for reproducibility.
        frac (float): Fraction of training data to use.

    Returns:
        float: Pearson correlation between predicted and true fitness values on validation set.
    """
    seed(42 + i)

    # Randomly subsample training data
    train_id = choice(range(train_full_data.shape[0]), int(train_full_data.shape[0] * frac))
    train_data = train_full_data[train_id, :]

    # Initialize additive latent model
    model_add_lantern = Add_Latent(
        data.nb_val,
        nb_state=hparam['nb_state'],
        emb=hparam['emb'],
        hid=hparam['hid'],
        nb_layer=hparam['nb_layer']
    )

    # Train the model
    _ = train(
        model_add_lantern,
        train_data,
        lr=hparam['lr'],
        wei_dec=hparam['wei_dec'],
        nb_epoch=hparam['nb_epoch'],
        bsize=hparam['bsize']
    )

    # Evaluate correlation on validation data
    fit_add_latern = model_add_lantern(val_data[:, :-1].long())[0].detach().squeeze(-1)
    cor = pearsonr(fit_add_latern, val_data[:, -1])[0]
    return cor


# ===== Run Experiments =====
max_iter = 30  # Number of independent runs
result = {}

print(f"Training {model_name} on {config.data_flag.upper()} dataset [{flag}]")
for frac in val_frac:
    print(f"\nTraining fraction: {frac:.3f}")
    tmp_w = []
    for i in tqdm(range(max_iter)):
        r = run_one(i, frac)
        tmp_w.append(r)
        print(f"Run {i+1}/{max_iter}: Pearson = {r:.3f}")
    result[frac] = tmp_w


# ===== Save Results =====
path_save = f'results/{config.data_flag}_{flag}'
os.makedirs(path_save, exist_ok=True)

joblib.dump(result, f"{path_save}/reg_{model_name}.joblib")

print(f"\n✅ Results saved to {path_save}/reg_{model_name}.joblib")
