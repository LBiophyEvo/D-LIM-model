"""
mevv_f.py
-----------------------
This script trains a model using **MAVE-NN** (Tareen et al., Genome Biology 2022).

Reference:
    Tareen, A., Kinney, J.B. "MAVE-NN: learning genotype–phenotype maps from multiplex assays of variant effect." 
    Genome Biology 23, 248 (2022). https://doi.org/10.1186/s13059-022-02661-7

Supported datasets:
    - "harry"
    - "elife"
    - "protein_inter"

This script evaluates the predictive power of MAVE-NN models on genotype–phenotype data, 
by training with different fractions of available data and computing the Pearson correlation
between predictions and true phenotypes on a held-out validation set.

Results are saved as:
    results/<data_flag>_<flag>/reg_MAVE-NN.joblib

Usage:
    python mevv_f.py --data_flag elife --flag subtle
"""

# ===== Imports =====
import sys
sys.path.append('../../')

import os
import joblib
import argparse
import pandas as pd
import numpy as np
import mavenn
import matplotlib.pyplot as plt

from numpy import logspace
from numpy.random import choice, seed
from scipy.stats import pearsonr
from tqdm import tqdm
from src.utils import Data_model


# ===== Argument Parser =====
parser = argparse.ArgumentParser(description='Train MAVE-NN model on genotype–phenotype data.')
parser.add_argument('--flag', type=str, default='subtle', help="Task type: 'fitness', 'epistasis', 'subtle', or 'strong'")
parser.add_argument('--data_flag', type=str, default='elife', help="Dataset name: 'harry', 'elife', or 'protein_inter'")
config = parser.parse_args()

flag = config.flag
model_name = 'MAVE-NN'

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

# ===== Load Data =====
data = Data_model(data_path, 2, sep=sep)

# MAVE-NN requires sequence-based inputs (string sequences)
# Assumes preprocessed `.dat` file exists in `../data/kemble_<flag>.dat`
data_df = pd.read_csv(f"../data/kemble_{flag}.dat", sep="\t")
L = len(data_df.loc[0, 'x'])  # sequence length
alphabet = ["A", "C", "G", "U", "T", "N"]

# ===== Set Random Seed =====
seed(42)

# ===== Split Data =====
val_frac = logspace(-1, 0.1, num=7)  # fractions of training data to test
val_id = choice(range(data.data.shape[0]), int(data.data.shape[0] * 0.3), replace=False)
train_full_id = [i for i in range(data.data.shape[0]) if i not in val_id]

train_full_data = data[train_full_id, :]
val_data = data[val_id, :]


# ===== Training Function =====
def run_one(i: int, frac: float) -> float:
    """
    Train and evaluate a MAVE-NN model using a subset of the training data.

    Args:
        i (int): Run index (used for random seed).
        frac (float): Fraction of available training data to use.

    Returns:
        float: Pearson correlation coefficient between predictions and true phenotypes.
    """
    seed(42 + i)

    # Sample training subset
    train_id = choice(train_full_id, int(len(train_full_id) * frac), replace=False)

    # Initialize MAVE-NN model
    model_mave = mavenn.Model(
        L=L,
        alphabet=alphabet,
        gpmap_type='additive',                # Additive genotype–phenotype map
        regression_type='GE',                 # Global epistasis regression
        ge_noise_model_type='SkewedT',        # Noise model
        ge_heteroskedasticity_order=2         # Heteroskedastic noise modeling
    )

    # Set training data
    model_mave.set_data(x=data_df['x'][train_id], y=data_df['y'][train_id])

    # Train the model
    model_mave.fit(
        learning_rate=1e-3,
        epochs=1000,
        batch_size=64,
        early_stopping=True,
        early_stopping_patience=25,
        verbose=False
    )

    # Evaluate on validation data
    fit_mave = model_mave.x_to_yhat(data_df["x"][val_id])
    cor_m = pearsonr(fit_mave, val_data[:, -1])[0]
    return cor_m


# ===== Main Training Loop =====
max_iter = 30
result = {}

print(f"\nStarting MAVE-NN experiments ({max_iter} runs per data fraction)...\n")

for frac in val_frac:
    print(f"Training with fraction = {frac:.2f}")
    correlations = []

    for i in tqdm(range(max_iter), desc=f"Frac {frac:.2f}"):
        r = run_one(i, frac)
        correlations.append(r)
        print(f"Run {i}: Pearson r = {r:.4f}")

    result[frac] = correlations


# ===== Save Results =====
save_dir = f"results/{config.data_flag}_{flag}"
os.makedirs(save_dir, exist_ok=True)
save_path = f"{save_dir}/reg_{model_name}.joblib"

joblib.dump(result, save_path)
print(f"\n✅ Results saved to: {save_path}")
