"""
dlim_train.py
-----------------------
This script trains the D-LIM (Deep Latent Interaction Model) on different datasets,
such as "harry", "elife", and "protein_inter". It compares the performance of 
the spectral and non-spectral variants of the D-LIM model.

Each experiment evaluates model performance using Pearson correlation 
on a held-out validation dataset.

The results for each dataset and condition are saved in:
    results/<data_flag>_<flag>/reg_dlim.joblib

Usage:
    python dlim_train.py --flag epistasis --data_flag harry

Arguments:
    --flag : str
        The type of experiment to run (e.g., 'fitness', 'epistasis', 'subtle', etc.)
    --data_flag : str
        The dataset to use ('harry', 'elife', or 'protein_inter')
"""

# ===== Imports =====
import sys
sys.path.append('../../')  # Allow import from project root

import os
import joblib
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from numpy.random import choice, seed

# Visualization
import matplotlib.pyplot as plt  

# Project imports
from dlim.model import DLIM 
from dlim.dataset import Data_model
from dlim.api import DLIM_API


# ===== Argument Parser =====
parser = argparse.ArgumentParser(description='Train D-LIM models with and without spectral regularization.')
parser.add_argument('--flag', type=str, default='fitness', help="Task flag, e.g., 'fitness', 'epistasis', 'subtle'")
parser.add_argument('--data_flag', type=str, default='harry', help="Dataset to use: 'harry', 'elife', or 'protein_inter'")
config = parser.parse_args()

flag = config.flag
model_name = 'dlim'


# ===== Hyperparameter Configuration =====
if config.data_flag == 'harry':
    nb_layer = 1 if config.flag == 'epistasis' else 0
    hparam = {
        'nb_state': 50,
        'emb': 1,
        'hid': 32,
        'nb_layer': nb_layer,
        'lr': 1e-3,
        'wei_dec': 1e-3,
        'nb_epoch': 400,
        'bsize': 32,
        'sep': ','
    }
    if flag == 'fitness':
        data_path = "../data/data_env_1.csv"
    elif flag == 'epistasis':
        data_path = "../data/data_epis_1.csv"

elif config.data_flag == 'elife':
    hparam = {
        'nb_state': 50,
        'emb': 6,
        'hid': 64,
        'nb_layer': 1,
        'lr': 1e-3,
        'wei_dec': 1e-3,
        'nb_epoch': 300,
        'bsize': 128,
        'sep': '@'
    }
    if flag == 'subtle':
        data_path = "../../data/elife/elife_data_subtle_env.csv"
    else:
        data_path = "../../data/elife/elife_data_strong_env.csv"

elif config.data_flag == 'protein_inter':
    hparam = {
        'nb_state': 50,
        'emb': 6,
        'hid': 64,
        'nb_layer': 1,
        'lr': 1e-3,
        'wei_dec': 1e-3,
        'nb_epoch': 300,
        'bsize': 256,
        'sep': ','
    }
    data_path = "../data/protein_inter/tables1.csv"

else:
    raise ValueError(f"Unknown dataset flag: {config.data_flag}")


# ===== Load and Prepare Data =====
df_data = pd.read_csv(data_path, sep=hparam['sep'], header=None)
data = Data_model(data=df_data, n_variables=2)

# Split into training and validation sets
seed(42)
val_id = choice(range(data.data.shape[0]), int(data.data.shape[0] * 0.3))
train_full_id = [i for i in range(data.data.shape[0]) if i not in val_id]
val_data = data.subset(val_id)
num_train_full_data = len(train_full_id)

# Define fractions of data to test training size effects
val_frac = [1.0] #logspace(-1, 0.1, num=7)  # Varying fractions of training data (10% to ~1.26×)


# ===== Training Function =====
def run_one(i, frac):
    """
    Train D-LIM models (with and without spectral regularization) on a random
    subset of the training data, then evaluate on the validation set.

    Args:
        i (int): Random seed offset for reproducibility across runs.
        frac (float): Fraction of training data to use.

    Returns:
        float: The best Pearson correlation achieved between model predictions and ground truth.
    """
    seed(42 + i)

    # Sample a random training subset
    train_id = choice(train_full_id, int(num_train_full_data * frac))
    train_data = data.subset(train_id)

    # ---- Model 1: D-LIM with Spectral Regularization ----
    model_spec = DLIM(n_variables=train_data.nb_val, hid_dim=hparam['hid'], nb_layer=hparam['nb_layer'])
    dlim_spec = DLIM_API(model=model_spec, flag_spectral=True)
    _ = dlim_spec.fit(
        train_data,
        lr=hparam['lr'],
        nb_epoch=hparam['nb_epoch'],
        batch_size=hparam['bsize'],
        emb_regularization=0,
        similarity_type='pearson'
    )

    fit_spec, _, _ = dlim_spec.predict(val_data.data[:, :-1], detach=True)
    cor_spec, _ = pearsonr(fit_spec.flatten(), val_data.data[:, [-1]].flatten())

    # ---- Model 2: D-LIM without Spectral Regularization ----
    model_no_spec = DLIM(n_variables=train_data.nb_val, hid_dim=hparam['hid'], nb_layer=hparam['nb_layer'])
    dlim_no_spec = DLIM_API(model=model_no_spec, flag_spectral=False)
    _ = dlim_no_spec.fit(
        train_data,
        lr=hparam['lr'],
        nb_epoch=hparam['nb_epoch'],
        batch_size=hparam['bsize'],
        emb_regularization=0,
        similarity_type='pearson'
    )

    fit_no_spec, _, _ = dlim_no_spec.predict(val_data.data[:, :-1], detach=True)
    cor_no_spec, _ = pearsonr(fit_no_spec.flatten(), val_data.data[:, [-1]].flatten())

    # Take the best-performing variant
    best_cor = max(cor_spec, cor_no_spec)
    return best_cor


# ===== Run Experiments =====
max_iter = 30  # Number of independent runs per fraction
result = {}

print(f"Training {model_name.upper()} model on {config.data_flag.upper()} dataset [{flag}]")

for frac in val_frac:
    print(f"\nTraining with {frac:.2f} fraction of data...")
    tmp_results = []
    for i in tqdm(range(max_iter)):
        r = run_one(i, frac)
        tmp_results.append(r)
        print(f"Run {i+1}/{max_iter}: Pearson = {r:.3f}")
    result[frac] = tmp_results


# ===== Save Results =====
path_save = f"results/{config.data_flag}_{flag}"
os.makedirs(path_save, exist_ok=True)
joblib.dump(result, f"{path_save}/reg_{model_name}.joblib")

print(f"\n✅ Results saved to {path_save}/reg_{model_name}.joblib")
