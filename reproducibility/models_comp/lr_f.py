"""
lr_f.py
-----------------------
This script trains a simple **Linear Regression (LR)** model as a baseline
for genotype–phenotype prediction tasks.

It supports different datasets such as:
    - "harry"
    - "elife"
    - "protein_inter"

The LR model provides a benchmark for evaluating the performance of 
more complex models such as ALM, DLIM, and LANTERN.

Results are saved as:
    results/<data_flag>_<flag>/reg_LR.joblib

Usage:
    python lr_f.py --data_flag elife --flag subtle
"""

# ===== Imports =====
import sys
sys.path.append('../../')

import os
import joblib
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from numpy.random import choice, seed
from numpy import logspace
from scipy.stats import pearsonr

from src.model import Regression
from src.utils import Data_model, train_reg


# ===== Argument Parser =====
parser = argparse.ArgumentParser(description='Train Linear Regression baseline model.')
parser.add_argument('--flag', type=str, default='subtle', help="Task type: 'fitness', 'epistasis', 'subtle', 'strong'")
parser.add_argument('--data_flag', type=str, default='elife', help="Dataset name: 'harry', 'elife', or 'protein_inter'")
config = parser.parse_args()

flag = config.flag
model_name = 'LR'

# ===== Dataset Configuration =====
if config.data_flag == 'harry':
    if flag == 'fitness':
        data_path = "../data/data_env_1.csv"
    elif flag == 'epistasis':
        data_path = "../data/data_epis_1.csv"
    sep = ','

elif config.data_flag == 'elife':
    if flag == 'subtle':
        data_path = "../data/elife/elife_data_subtle_env.csv"
    else:
        data_path = "../data/elife/elife_data_strong_env.csv"
    sep = '@'

elif config.data_flag == 'protein_inter':
    data_path = "../data/protein_inter/tables1.csv"
    sep = ','

else:
    raise ValueError(f"Unknown dataset flag: {config.data_flag}")

print(f"\nTraining {model_name} model on dataset '{config.data_flag}' ({flag})")
print(f"Data path: {data_path}")

# ===== Set Random Seed for Reproducibility =====
seed(42)

# ===== Load Data =====
data = Data_model(data_path, n_variables=2, sep=sep)

# Define validation and training splits
val_frac = logspace(-1, 0.1, num=7)  # Training fractions (log scale)
val_id = choice(range(data.data.shape[0]), int(data.data.shape[0] * 0.3), replace=False)
train_full_id = [i for i in range(data.data.shape[0]) if i not in val_id]

train_full_data = data[train_full_id, :]
val_data = data[val_id, :]


# ===== Training Function =====
def run_one(seed_offset: int, frac: float) -> float:
    """
    Train and evaluate a linear regression model using a subset of the data.

    Args:
        seed_offset (int): Random seed offset for reproducibility.
        frac (float): Fraction of available training data to use.

    Returns:
        float: Pearson correlation between predictions and true values on validation set.
    """
    seed(42 + seed_offset)
    
    # Randomly select training subset
    train_id = choice(range(train_full_data.shape[0]), int(train_full_data.shape[0] * frac), replace=False)
    train_data = train_full_data[train_id, :]

    # Initialize linear regression model
    model_reg = Regression(nb_var=data.nb_val, nb_state=37)

    # Train the model
    _ = train_reg(model_reg, train_data, lr=1e-2, nb_epoch=300, bsize=64)

    # Evaluate on validation data
    fit_reg = model_reg(val_data[:, :-1].long()).detach().squeeze(-1)
    cor = pearsonr(fit_reg, val_data[:, -1])[0]
    return cor


# ===== Experiment Configuration =====
max_iter = 30  # number of repetitions per data fraction
result = {}

print(f"\nStarting Linear Regression experiments with {max_iter} runs per data fraction...\n")

# ===== Main Training Loop =====
for frac in val_frac:
    print(f"Training with fraction = {frac:.2f}")
    correlations = []
    
    for i in tqdm(range(max_iter), desc=f"Frac {frac:.2f}"):
        r = run_one(i, frac)
        correlations.append(r)
    
    result[frac] = correlations

# ===== Save Results =====
save_dir = f"results/{config.data_flag}_{flag}"
os.makedirs(save_dir, exist_ok=True)
save_path = f"{save_dir}/reg_{model_name}.joblib"

joblib.dump(result, save_path)
print(f"\n✅ Results saved to: {save_path}")
