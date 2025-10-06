"""
Train and compare DLIM (Deep Latent Interaction Model) models 
on *simulated data* for different data types (cascade, bio, tgaus).

This script:
1. Generates simulated data using the specified model type.
2. Trains multiple DLIM models (with and without spectral regularization).
3. Collects learned embeddings from each run.
4. Computes cosine similarity between embeddings across runs.
5. Plots the similarity distribution to evaluate training stability.

Usage:
    python get_similarity_simulated.py --data_flag cascade

"""

# ======= Imports =======
import sys
sys.path.append('../../')  # Ensure parent directory is importable

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm  
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from numpy.random import choice
from numpy import mean

# Project-specific imports
from dlim.model import DLIM 
from dlim.dataset import Data_model
from dlim.api import DLIM_API
from src_simulate_data.sim_data import Simulated


# ======= Argument Parsing =======
parser = argparse.ArgumentParser(description='Train DLIM models on simulated data.')
parser.add_argument(
    '--data_flag',
    type=str,
    default='cascade',
    help="Type of simulated data to generate: 'cascade', 'bio', or 'tgaus'."
)

config = parser.parse_args()
type_f = config.data_flag

# ======= Directory Setup =======
model_save_abs_path = f'pretrained_model/{type_f}/'
if not os.path.exists(model_save_abs_path):
    os.makedirs(model_save_abs_path)


# ======= Simulated Data Generation =======
nb_var = 30  # number of variables/features to simulate
data_simulated = Simulated(nb_var, type_f)

# Wrap into Data_model (for DLIM compatibility)
data = Data_model(data=pd.DataFrame(data_simulated.data), n_variables=2)


# ======= Model Training =======
run_iter = 30            # Number of independent training runs
tmp = []                 # Store embeddings from all runs
best_loss_arr = []       # (Optional) to track best losses if needed

for i in range(run_iter):
    # Ensure reproducibility
    torch.manual_seed(42 + i)
    random.seed(42 + i)

    # --- DLIM with Spectral Regularization ---
    model = DLIM(
        n_variables=data.nb_val,
        hid_dim=32,
        nb_layer=0,
        gap_thres=[0.01, 0.95]
    )
    model_save_path = os.path.join(model_save_abs_path, f"{type_f}_spec{i}.pt")
    dlim_regressor = DLIM_API(model=model, flag_spectral=True, load_model=model_save_path)

    # --- DLIM without Spectral Regularization ---
    model_ns = DLIM(
        n_variables=data.nb_val,
        hid_dim=32,
        nb_layer=0,
        gap_thres=[0.01, 0.95]
    )
    model_save_path = os.path.join(model_save_abs_path, f"{type_f}_no_spec{i}.pt")
    dlim_regressor_ns = DLIM_API(model=model_ns, flag_spectral=False, load_model=model_save_path)

    # --- Extract learned embeddings ---
    emb_with_spec = (
        dlim_regressor.model.genes_emb[0].detach().numpy(),
        dlim_regressor.model.genes_emb[1].detach().numpy()
    )
    emb_no_spec = (
        dlim_regressor_ns.model.genes_emb[0].detach().numpy(),
        dlim_regressor_ns.model.genes_emb[1].detach().numpy()
    )

    tmp.append((emb_with_spec, emb_no_spec))


# ======= Cosine Similarity Function =======
def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors or matrices.
    Both inputs are flattened before computation.

    Args:
        a (np.ndarray): First vector or matrix.
        b (np.ndarray): Second vector or matrix.

    Returns:
        float: Cosine similarity between `a` and `b`.
    """
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ======= Compute Similarities Across Runs =======
dist_l = []
for i, ((g1i, g2i), _) in enumerate(tmp):
    # Compare embeddings between different runs
    for j, ((g1j, g2j), _) in enumerate(tmp[i + 1:], start=i + 1):
        dist_l.append(abs(cosine_similarity(g1i, g1j)))


# ======= Plot Similarity Distribution =======
fig, ax = plt.subplots(figsize=(2, 2))
ax.hist(dist_l, bins=20, color='skyblue', edgecolor='black')

# Style adjustments
for el in ["top", "right"]:
    ax.spines[el].set_visible(False)

ax.set_xlabel("$\\rho$ (Cosine Similarity)", fontsize=8)
ax.set_ylabel("Count (#N)", fontsize=8)
plt.tight_layout()

# Save and show plot
if not os.path.exists("figures"):
    os.makedirs("figures")

plt.savefig(f"figures/similarity_{type_f}.png")
plt.show()
