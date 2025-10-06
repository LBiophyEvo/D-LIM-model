"""
stability_simulated.py
-----------------------

This script trains multiple DLIM (Deep Latent Interaction Model) instances 
on simulated datasets with and without spectral regularization. 
It evaluates model performance in terms of validation MSE and Pearson correlation 
and generates comparative plots for analysis.

Usage:
    python train_dlim_simulated.py --data_flag cascade

Arguments:
    --data_flag : str
        Type of simulated dataset to use ('cascade', 'bio', 'tgaus', etc.)

Outputs:
    - Saved pretrained models in: ./pretrained_model/<data_flag>/
    - Figures in: ./figures/
        * convergence_<data_flag>.png
        * conv_hist_<data_flag>.png
        * boxplot_<data_flag>_mse.png
        * boxplot_<data_flag>_pearson.png
"""

# ===== Imports =====
import sys
sys.path.append('../../')  # Add project root to Python path

import os
import random
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.random import choice
from numpy import mean
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# Custom project modules
from dlim.model import DLIM
from dlim.dataset import Data_model
from dlim.api import DLIM_API
from src_simulate_data.sim_data import Simulated


# ===== Argument Parsing =====
parser = argparse.ArgumentParser(description='Train DLIM models on simulated data.')
parser.add_argument('--data_flag', type=str, default='cascade',
                    help="Type of simulation to use ('cascade', 'bio', 'tgaus').")
config = parser.parse_args()

type_f = config.data_flag  # Simulation type flag (dataset structure)
model_save_abs_path = f'pretrained_model/{type_f}/'

# Create directory for saving models
os.makedirs(model_save_abs_path, exist_ok=True)


# ===== Data Simulation =====
nb_var = 30  # Number of variables (features) in the simulated data
data_simulated = Simulated(nb_var, type_f)  # Generate simulated dataset
data = Data_model(data=pd.DataFrame(data_simulated.data), n_variables=2)  # Wrap into DLIM dataset format


# ===== Training Configuration =====
run_iter = 30  # Number of runs (random restarts)
best_loss_arr = []  # To store results across runs
tmp = []  # To store loss curves for plotting convergence


# ===== Training Loop =====
for i in range(run_iter):
    random.seed(42 + i)
    torch.manual_seed(42 + i)

    # Split data into training and validation sets
    train_id = choice(range(data.data.shape[0]), int(data.data.shape[0] * 0.7), replace=False)
    val_id = [j for j in range(data.data.shape[0]) if j not in train_id]
    train_data = data.subset(train_id)
    val_data = data.subset(val_id)

    # ---------------------------
    # Train DLIM with Spectral Regularization
    # ---------------------------
    model = DLIM(n_variables=data.nb_val, hid_dim=32, nb_layer=0, gap_thres=[0.01, 0.95])
    dlim_regressor = DLIM_API(model=model, flag_spectral=True)
    model_save_path = f"{model_save_abs_path}{type_f}_spec{i}.pt"

    # Train model
    losses, best_model, best_loss = dlim_regressor.fit(
        train_data, val_data,
        lr=1e-3, nb_epoch=600, batch_size=64,
        emb_regularization=0, save_path=model_save_path,
        max_patience=20, return_best_model=True
    )

    # Evaluate Pearson correlation on validation set
    fit_a, var_a, lat_a = dlim_regressor.predict(val_data.data[:, :-1], detach=True)
    score = pearsonr(fit_a.flatten(), val_data.data[:, [-1]].flatten())[0]
    print(f"[Run {i}] Pearson (Spectral): {score:.3f}")

    # ---------------------------
    # Train DLIM without Spectral Regularization
    # ---------------------------
    model_ns = DLIM(n_variables=data.nb_val, hid_dim=32, nb_layer=0, gap_thres=[0.01, 0.95])
    dlim_regressor_ns = DLIM_API(model=model_ns, flag_spectral=False)
    model_save_path = f"{model_save_abs_path}{type_f}_no_spec{i}.pt"

    losses_ns, best_model_ns, best_loss_ns = dlim_regressor_ns.fit(
        train_data, val_data,
        lr=1e-3, nb_epoch=600, batch_size=64,
        emb_regularization=0, save_path=model_save_path,
        max_patience=20, return_best_model=True
    )

    # Evaluate Pearson correlation for non-spectral model
    fit_ns, var_ns, lat_ns = dlim_regressor_ns.predict(val_data.data[:, :-1], detach=True)
    score_ns = pearsonr(fit_ns.flatten(), val_data.data[:, [-1]].flatten())[0]
    print(f"[Run {i}] Pearson (No Spectral): {score_ns:.3f}")

    # Extract validation losses for convergence plot
    _, vl = zip(*losses)
    _, vl_ns = zip(*losses_ns)
    tmp.append((vl, vl_ns))

    # Store final metrics
    best_loss_arr.append((best_loss, best_loss_ns, score, score_ns))


# ===== Visualization Section =====

# --- Plot training convergence ---
fig, ax = plt.subplots(figsize=(2, 2))
for vl, vl_ns in tmp:
    ax.plot(vl, linestyle="-", c="orange", alpha=0.7, label='Spec.' if vl == tmp[0][0] else "")
    ax.plot(vl_ns, linestyle="-", c="grey", alpha=0.7, label='No Spec.' if vl == tmp[0][0] else "")
for el in ["top", "right"]:
    ax.spines[el].set_visible(False)
ax.set_xlabel("Epoch", fontsize=8)
ax.set_ylabel("Validation MSE", fontsize=8)
ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig(f"./figures/convergence_{type_f}.png", dpi=300, transparent=True)
plt.show()


# --- Plot histogram of best MSE ---
vl_last, vl_ns_last, pears_last, pears_no_last = zip(*best_loss_arr)
print(f"Mean MSE (Spec.): {mean(vl_last):.4f} | (No Spec.): {mean(vl_ns_last):.4f}")

fig, ax = plt.subplots(figsize=(2, 2))
ax.hist([vl_last, vl_ns_last], color=["orange", "grey"], label=["Spec.", "No Spec."])
for el in ["top", "right"]:
    ax.spines[el].set_visible(False)
ax.set_xlabel("Best MSE", fontsize=8)
ax.set_ylabel("Frequency", fontsize=8)
ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig(f"./figures/conv_hist_{type_f}.png", dpi=300, transparent=True)
plt.show()


# --- Boxplot: Best MSE comparison ---
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
plt.savefig(f"./figures/boxplot_{type_f}_mse.png", dpi=300, transparent=True)
plt.show()


# --- Boxplot: Pearson correlation comparison ---
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
plt.savefig(f"./figures/boxplot_{type_f}_pearson.png", dpi=300, transparent=True)
plt.show()
