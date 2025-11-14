"""
lan.py
-----------------------
This script trains the LANTERN model on different datasets such as "harry", "elife", 
and "protein_inter".

LANTERN is an interpretable model for genotype–phenotype landscapes introduced in:
    Tonner, Peter D., Abe Pressman, and David Ross. 
    "Interpretable modeling of genotype–phenotype landscapes with state-of-the-art predictive power."
    *Proceedings of the National Academy of Sciences* 119.26 (2022): e2114021119.

The script evaluates model performance across different fractions of available training data,
using Pearson correlation between predicted and actual phenotypes on a held-out test set.

Results are saved to:
    results/<data_flag>_<flag>/reg_LANTERN.joblib

Usage:
    python lan.py --data_flag elife --flag subtle
"""

# ===== Imports =====
import sys
sys.path.append('../../')

import os
import joblib
import torch
import argparse
import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm
from scipy.stats import pearsonr
from numpy.random import choice, seed
from numpy import logspace
from torch.optim import Adam

from lantern.model.basis import VariationalBasis
from lantern.model.surface import Phenotype
from lantern.model import Model
from lantern.model.likelihood import GaussianLikelihood
from lantern.dataset import Dataset


# ===== Utility Function =====
def read_data(file: str, nb_var: int = 2, sep: str = ',') -> pd.DataFrame:
    """
    Read genotype-phenotype data and format it for LANTERN.

    Args:
        file (str): Path to the CSV data file.
        nb_var (int): Number of genotype columns before the phenotype column.
        sep (str): Column separator used in the file.

    Returns:
        pd.DataFrame: Formatted dataframe with 'substitutions' and 'phenotype' columns.
    """
    data = pd.read_csv(file, header=None, sep=sep).dropna()
    fit = data.iloc[:, nb_var]
    comb = data.iloc[:, 0].astype(str) + '_gene1:' + data.iloc[:, 1].astype(str) + '_gene2'
    df = pd.DataFrame({'substitutions': comb, 'phenotype': fit})
    return df


# ===== Parallel Training Class =====
class ParallelLanternRun:
    """
    Class for running LANTERN model training and evaluation for a given dataset subset.
    """
    def __init__(self, df, train_full_id, Xtest, ytest):
        self.df = df
        self.train_full_id = train_full_id
        self.Xtest = Xtest
        self.ytest = ytest

    def run_one_lantern(self, i: int, frac: float) -> float:
        """
        Train and evaluate a LANTERN model on a random subset of the training data.

        Args:
            i (int): Random seed offset.
            frac (float): Fraction of training data to use.

        Returns:
            float: Pearson correlation between predicted and true phenotypes.
        """
        seed(42 + i)

        # Select subset of training data
        len_train = len(self.train_full_id)
        train_id = choice(self.train_full_id, int(len_train * frac), replace=False)
        left_id = [idx for idx in self.train_full_id if idx not in train_id]

        # Create dataset with missing values for validation points
        df_copy = self.df.copy()
        df_copy.loc[left_id, 'phenotype'] = np.nan  # safer assignment
        ds = Dataset(df_copy)

        # Define LANTERN model components
        K = 4  # latent dimension
        basis = VariationalBasis.fromDataset(ds, K=K)
        surface = Phenotype.fromDataset(ds, K=K, Ni=200, inducScale=1.0)
        model = Model(basis, surface, GaussianLikelihood())
        loss_fn = model.loss(N=len(ds))

        # Prepare training data
        Xtrain, ytrain = ds[train_id]
        optimizer = Adam(loss_fn.parameters(), lr=0.01)

        # Training loop
        epochs = 1000
        hist = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            yhat = model(Xtrain)
            losses = loss_fn(yhat, ytrain)
            total_loss = sum(losses.values())
            total_loss.backward()
            optimizer.step()
            hist.append(total_loss.item())

        # Evaluate model
        with torch.no_grad():
            Z_t = model.basis(self.Xtest)
            yhat = model.surface(Z_t)
            y_pred = yhat.mean.detach().numpy()

        pearson_corr = pearsonr(self.ytest.detach().numpy().flatten(), y_pred.flatten())[0]
        return pearson_corr


# ===== Main Script =====
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LANTERN models.')
    parser.add_argument('--flag', type=str, default='subtle', help="Task type, e.g., 'subtle', 'strong', 'fitness'")
    parser.add_argument('--data_flag', type=str, default='elife', help="Dataset name: 'harry', 'elife', or 'protein_inter'")
    config = parser.parse_args()

    flag = config.flag
    model_name = 'LANTERN'

    # ----- Dataset Selection -----
    if config.data_flag == 'harry':
        if flag == 'fitness':
            data_path = "../data/data_env_1.csv"
        elif flag == 'epistasis':
            data_path = "../data/data_epis_1.csv"
        sep = ','

    elif config.data_flag == 'elife':
        sep = '@'
        if flag == 'subtle':
            data_path = "../../data/elife/elife_data_subtle_env.csv"
        else:
            data_path = "../../data/elife/elife_data_strong_env.csv"

    elif config.data_flag == 'protein_inter':
        sep = ','
        data_path = "../data/protein_inter/tables1.csv"

    else:
        raise ValueError(f"Unknown dataset flag: {config.data_flag}")

    # ----- Read and Prepare Data -----
    df = read_data(file=data_path, sep=sep)
    seed(42)
    val_frac = [1.0] #logspace(-1, 0.1, num=7)  # Varying fractions of training data (10% to ~1.26×)
    val_id = choice(range(df.shape[0]), int(df.shape[0] * 0.3), replace=False)
    train_full_id = [i for i in range(df.shape[0]) if i not in val_id]

    ds_copy = Dataset(df)
    Xtest, ytest = ds_copy[val_id]
    df.loc[val_id, 'phenotype'] = np.nan  # Mark validation set as missing

    # Initialize runner
    run_iter = ParallelLanternRun(df, train_full_id, Xtest, ytest)

    # ----- Train Model -----
    print(f"\nTraining {model_name} on {config.data_flag.upper()} dataset [{flag}]")
    result = {}
    start_time = time()

    for frac in val_frac:
        print(f"\nTraining with {frac:.2f} fraction of data...")
        corr_list = []
        for i_time in tqdm(range(30)):
            corr = run_iter.run_one_lantern(i_time, frac)
            corr_list.append(corr)
        result[frac] = corr_list

    end_time = time()
    print(f"\n✅ LANTERN training completed in {end_time - start_time:.2f} seconds")

    # ----- Save Results -----
    save_dir = f"results/{config.data_flag}_{flag}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/reg_{model_name}.joblib"
    joblib.dump(result, save_path)
    print(f"Results saved to {save_path}")
