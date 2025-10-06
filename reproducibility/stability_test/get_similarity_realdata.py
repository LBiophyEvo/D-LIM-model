"""
Train and compare DLIM (Deep Latent Interaction Model) models 
on different datasets and environments.

This script:
1. Loads data for a given dataset/environment combination. This script allows for data from Kemble et al. and Kinsler et al.
2. Trains multiple instances of DLIM with and without spectral regularization.
3. Computes embedding similarities across runs.
4. Plots the cosine similarity distribution of embeddings.

Usage:
    python get_similarity_realdata.py --data_flag harry --env env_1
"""

# ======= Imports =======
import sys
sys.path.append('../../')  # Add project root to Python path for imports
import os
import random
import argparse

# Core libraries
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Progress tracking
from tqdm import tqdm  

# Statistical & ML utilities
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from numpy import mean
from numpy.random import choice

# Project-specific modules
from dlim.model import DLIM 
from dlim.dataset import Data_model
from dlim.api import DLIM_API
from src_simulate_data.sim_data import Simulated


# ======= Argument Parsing =======
parser = argparse.ArgumentParser(description='Train all DLIM models with and without spectral regularization.')

parser.add_argument('--data_flag', type=str, default='harry',
                    help="Dataset to use: 'harry' or 'elife'")
parser.add_argument('--env', type=str, default='env_1',
                    help="Environment setting (e.g., 'env_1', 'env_2', 'epis_1', 'subtle', 'strong')")

config = parser.parse_args()
type_f = config.data_flag
env = config.env 

# ======= Directory Setup =======
save_name = f"{type_f}_{env}"
model_save_abs_path = f'pretrained_model/{type_f}/{env}/'

if not os.path.exists(model_save_abs_path):
    os.makedirs(model_save_abs_path)


# ======= Data Loading and Hyperparameters =======
if type_f == 'harry':
    df_data = pd.read_csv(f"../data/data_{env}.csv", sep=',', header=None)
    hparam = {
        'hid_dim': 32,
        'nb_layer': 0,
        'lr': 1e-2, 
        'weight_decay': 1e-4,
        'nb_epoch': 600, 
        'batch_size': 64,
        'emb_regularization': 0, 
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
        'emb_regularization': 1e-2
    }

# Wrap data into Data_model
data = Data_model(data=df_data, n_variables=2)

# List all mutations (excluding WT)
all_mut_1 = [n for n in data.substitutions_tokens[0].keys() if n != "WT"]
all_mut_2 = [n for n in data.substitutions_tokens[1].keys() if n != "WT"]

# ======= Model Training Loop =======
run_iter = 30  # number of repeated runs
tmp = []       # store embeddings from each run

for i in range(run_iter):
    random.seed(42 + i)
    
    # Split into train/validation sets
    train_id = choice(range(data.data.shape[0]), int(data.data.shape[0] * 0.7), replace=False)
    val_id = [j for j in range(data.data.shape[0]) if j not in train_id]

    train_data = data.subset(train_id)
    val_data = data.subset(val_id)

    # ------- Train Model with Spectral Regularization -------
    torch.manual_seed(42 + i)
    model = DLIM(
        n_variables=data.nb_val,
        hid_dim=hparam['hid_dim'],
        nb_layer=hparam['nb_layer'],
        gap_thres=[0.01, 0.95]
    )
    model_save_path = os.path.join(model_save_abs_path, f"{type_f}_spec{i}.pt")
    dlim_regressor = DLIM_API(model=model, flag_spectral=True, load_model=model_save_path)

    # ------- Train Model without Spectral Regularization -------
    model_ns = DLIM(
        n_variables=data.nb_val,
        hid_dim=32,
        nb_layer=0,
        gap_thres=[0.01, 0.95]
    )
    model_save_path = os.path.join(model_save_abs_path, f"{type_f}_no_spec{i}.pt")
    dlim_regressor_ns = DLIM_API(model=model_ns, flag_spectral=False, load_model=model_save_path)

    # ------- Extract Embeddings -------
    # For both models (with and without spectral regularization)
    araa_d = [dlim_regressor.model.genes_emb[0][data.substitutions_tokens[0][n]].detach().numpy() for n in all_mut_1]
    arab_d = [dlim_regressor.model.genes_emb[1][data.substitutions_tokens[1][n]].detach().numpy() for n in all_mut_2]

    araa_d_ns = [dlim_regressor_ns.model.genes_emb[0][data.substitutions_tokens[0][n]].detach().numpy() for n in all_mut_1]
    arab_d_ns = [dlim_regressor_ns.model.genes_emb[1][data.substitutions_tokens[1][n]].detach().numpy() for n in all_mut_2]

    # Store embeddings from current iteration
    res = ((araa_d, arab_d), ())
    tmp.append(res)


# ======= Similarity Computation =======
def cosine_similarity(a, b):
    """
    Compute cosine similarity between two flattened vectors.
    """
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Compute pairwise cosine similarities between embeddings across runs
dist_l = []
for i, ((g1i, g2i), _) in enumerate(tmp):
    for j, ((g1j, g2j), _) in enumerate(tmp[i + 1:], start=i + 1):
        dist_l.append(abs(cosine_similarity(g1i, g1j)))

# ======= Visualization =======
fig, ax = plt.subplots(figsize=(2, 2))
ax.hist(dist_l, bins=20, color='skyblue', edgecolor='black')

# Clean up plot style
for el in ["top", "right"]:
    ax.spines[el].set_visible(False)

ax.set_xlabel("$\\rho$ (Cosine Similarity)", fontsize=8)
ax.set_ylabel("Count (#N)", fontsize=8)
plt.tight_layout()

# Save and show plot
plt.savefig(f"figures/similarity_{save_name}.png")
plt.show()
