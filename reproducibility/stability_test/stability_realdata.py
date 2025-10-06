"""
Train and evaluate multiple DLIM (Deep Latent Interaction Model) instances 
in parallel with and without spectral regularization.

This script:
1. Loads real datasets (e.g., 'harry', 'elife') for a given environment.
2. Trains multiple DLIM models using multiprocessing.
3. Compares training performance (MSE loss) and Pearson correlation scores
   between spectral and non-spectral models.
4. Visualizes results with histograms and boxplots.

Usage:
    python stability_realdata.py --data_flag harry --env env_1

    python stability_realdata.py --data_flag elife --env strong

"""

# ===== Imports =====
import sys
sys.path.append('../../')  # Allow importing local project modules

import os
import random
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.random import choice
from numpy import mean
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# Project modules
from dlim.model import DLIM
from dlim.dataset import Data_model
from dlim.api import DLIM_API
from src_simulate_data.sim_data import Simulated


# ===== Argument Parsing =====
parser = argparse.ArgumentParser(description='Train multiple DLIM models in parallel.')
parser.add_argument('--data_flag', type=str, default='harry',
                    help="Dataset type: 'harry' or 'elife'")
parser.add_argument('--env', type=str, default='env_1',
                    help="Environment setting (e.g., 'env_1', 'env_2', 'epis_1', 'strong', 'subtle')")

config = parser.parse_args()
type_f = config.data_flag
env = config.env

save_name = f"{type_f}_{env}"
model_save_abs_path = f"pretrained_model/{type_f}/{env}/"

# Create directory for saving models if not existing
os.makedirs(model_save_abs_path, exist_ok=True)


# ===== Data Loading =====
if type_f == 'harry':
    df_data = pd.read_csv(f"../data/data_{env}.csv", sep=',', header=None)
    hparam = {
        'hid_dim': 32,
        'nb_layer': 0,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'nb_epoch': 300,
        'batch_size': 64,
        'emb_regularization': 0,
        'max_patience': 100,
    }
elif type_f == 'elife':
    df_data = pd.read_csv(f"../data/elife/elife_data_{env}_env.csv", sep='@', header=None)
    hparam = {
        'hid_dim': 64,
        'nb_layer': 1,
        'lr': 1e-3,
        'weight_decay': 1e-3,
        'nb_epoch': 200,
        'batch_size': 128,
        'emb_regularization': 1e-2,
        'max_patience': 100,
    }
else:
    raise ValueError(f"Unknown data_flag: {type_f}")

# Wrap dataset for DLIM
data = Data_model(data=df_data, n_variables=2)

# ===== Train/Validation Split =====
train_id = choice(range(data.data.shape[0]), int(data.data.shape[0] * 0.7), replace=False)
val_id = [i for i in range(data.data.shape[0]) if i not in train_id]
train_data = data.subset(train_id)
val_data = data.subset(val_id)


# ===== Function: Train One Pair of Models =====
def run_one(i):
    """
    Train a pair of DLIM models (with and without spectral regularization)
    and evaluate their performance.

    Args:
        i (int): Iteration index (used for random seeds and logging)

    Returns:
        tuple: (best_loss_spec, best_loss_no_spec, pearson_spec, pearson_no_spec)
    """
    random.seed(42 + i)
    torch.manual_seed(42 + i)

    # --- Spectral Regularization Model ---
    model = DLIM(
        n_variables=data.nb_val,
        hid_dim=hparam['hid_dim'],
        nb_layer=hparam['nb_layer'],
        gap_thres=[0.01, 0.95],
    )
    dlim_regressor = DLIM_API(model=model, flag_spectral=True)

    losses, best_model, best_loss = dlim_regressor.fit(
        train_data,
        val_data,
        lr=hparam['lr'],
        weight_decay=hparam['weight_decay'],
        nb_epoch=hparam['nb_epoch'],
        batch_size=hparam['batch_size'],
        emb_regularization=hparam['emb_regularization'],
        save_path=None,
        max_patience=hparam['max_patience'],
        return_best_model=True,
    )

    # Predict and compute Pearson correlation
    fit_a, var_a, lat_a = dlim_regressor.predict(val_data.data[:, :-1], detach=True)
    score = pearsonr(fit_a.flatten(), val_data.data[:, [-1]].flatten())[0]

    # --- Non-Spectral Regularization Model ---
    model_ns = DLIM(
        n_variables=data.nb_val,
        hid_dim=hparam['hid_dim'],
        nb_layer=hparam['nb_layer'],
        gap_thres=[0.01, 0.95],
    )
    dlim_regressor_ns = DLIM_API(model=model_ns, flag_spectral=False)

    losses_ns, best_model_ns, best_loss_ns = dlim_regressor_ns.fit(
        train_data,
        val_data,
        lr=hparam['lr'],
        weight_decay=hparam['weight_decay'],
        nb_epoch=hparam['nb_epoch'],
        batch_size=hparam['batch_size'],
        emb_regularization=hparam['emb_regularization'],
        save_path=None,
        max_patience=hparam['max_patience'],
        return_best_model=True,
    )

    fit_ns, var_ns, lat_ns = dlim_regressor_ns.predict(val_data.data[:, :-1], detach=True)
    score_ns = pearsonr(fit_ns.flatten(), val_data.data[:, [-1]].flatten())[0]

    print(f"[Run {i}] Pearson (Spec/NoSpec): {score:.3f} / {score_ns:.3f}")

    return best_loss, best_loss_ns, score, score_ns


# ===== Parallel Execution =====
run_iter = 30
with Pool() as pool:
    best_loss_arr = pool.map(run_one, range(run_iter))

# Unpack results
vl_last, vl_ns_last, pears_last, pears_no_last = zip(
    *[(vl_min, vl_ns_min, pears, pears_no)
      for vl_min, vl_ns_min, pears, pears_no in best_loss_arr]
)

print(f"\nMean Best MSE: Spec={mean(vl_last):.4f}, NoSpec={mean(vl_ns_last):.4f}")
print(f"Mean Pearson:  Spec={mean(pears_last):.4f}, NoSpec={mean(pears_no_last):.4f}")


# ===== Visualization =====

# --- Histogram of Best MSE ---
fig, ax = plt.subplots(figsize=(2, 2))
ax.hist([vl_last, vl_ns_last], color=["orange", "grey"], label=["Spec.", "No Spec."])
for el in ["top", "right"]:
    ax.spines[el].set_visible(False)
ax.set_xlabel("Best MSE", fontsize=8)
ax.set_ylabel("Frequency", fontsize=8)
ax.legend(fontsize=6)
plt.tight_layout()
# plt.savefig(f"./figures/conv_hist_{save_name}.png", dpi=300, transparent=True)

# --- Boxplot: Best MSE Comparison ---
fig, ax = plt.subplots(figsize=(2, 2))
bplot = ax.boxplot([vl_last, vl_ns_last], patch_artist=True)
ax.set_xticklabels(['Spec.', 'No Spec.'])
for patch, color in zip(bplot['boxes'], ["orange", "grey"]):
    patch.set_facecolor(color)
for patch in bplot['medians']:
    patch.set_color('black')
for el in ["top", "right"]:
    ax.spines[el].set_visible(False)
ax.set_xlabel("Models", fontsize=8)
ax.set_ylabel("Best MSE", fontsize=8)
plt.tight_layout()
# plt.savefig(f"./figures/boxplot_{save_name}_mse.png", dpi=300, transparent=True)

# --- Boxplot: Pearson Correlation Comparison ---
fig, ax = plt.subplots(figsize=(2, 2))
bplot = ax.boxplot([pears_last, pears_no_last], patch_artist=True)
ax.set_xticklabels(['Spec.', 'No Spec.'])
for patch, color in zip(bplot['boxes'], ["orange", "grey"]):
    patch.set_facecolor(color)
for patch in bplot['medians']:
    patch.set_color('black')
for el in ["top", "right"]:
    ax.spines[el].set_visible(False)
ax.set_xlabel("Models", fontsize=8)
ax.set_ylabel("$\\rho$ (Pearson)", fontsize=8)
plt.tight_layout()
# plt.savefig(f"./figures/boxplot_{save_name}_pearson.png", dpi=300, transparent=True)

plt.show()
