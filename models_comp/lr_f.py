import sys 
sys.path.append('../')
from src.model import  Regression, Add_Latent
from src.utils import Data_model, train, train_reg
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from numpy import mean, logspace, std
from numpy.random import choice, seed
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd
# from multiprocessing.pool import ThreadPool as Pool
# packages for lantern 
import torch 
import numpy as np 
import argparse
from tqdm import tqdm 
import joblib 
parser = argparse.ArgumentParser(description='Train all models')
parser.add_argument('--flag', type=str, default='subtle')
parser.add_argument('--data_flag', type=str, default='elife')

config = parser.parse_args()
flag =  config.flag

if config.data_flag == 'harry':
    if flag == 'fitness':
        data_path = "../data/data_env_1.csv"
    elif flag == 'epistasis':
        data_path = "../data/data_epis_1.csv"
elif config.data_flag == 'elife':
    if flag == 'subtle':
        data_path = "../data/elife/elife_data_subtle_env.csv"
    else:
        data_path = "../data/elife/elife_data_strong_env.csv"
elif config.data_flag == 'protein_inter':
    data_path = "../data/protein_inter/tables1.csv"
model_name = 'LANTERN'
if config.data_flag == 'elife':
    sep = '@'
else:
    sep = ','
model_name = 'LR'
print(flag)

seed(42)
data = Data_model(data_path, 2, sep=sep)
val_frac = logspace(-1, 0.1, num=7)
val_id = choice(range(data.data.shape[0]), int(data.data.shape[0]*0.3))
train_full_id = [i for i in range(data.data.shape[0]) if i not in val_id]
train_full_data = data[train_full_id, :]
val_data = data[val_id, :]

def run_one(i, frac):
    seed(42 + i)
    train_id = choice(range(train_full_data.shape[0]), int(train_full_data.shape[0]*frac))
    train_data = train_full_data[train_id, :]
    
    model_reg = Regression(nb_var=data.nb_val, nb_state=37)
    _ = train_reg(model_reg, train_data, lr=1e-2, nb_epoch=300, bsize=64)
    fit_reg = model_reg(val_data[:, :-1].long()).detach().squeeze(-1)
    cor = pearsonr(fit_reg, val_data[:, -1])[0]
    return cor 

max_iter = 30 
result = {}

print(model_name)
for frac in val_frac:
    print(frac)
    tmp_w = []
    for i in tqdm(range(max_iter)):
        r = run_one(i, frac) 
        tmp_w.append(r)
    result[frac] = tmp_w
joblib.dump(result, 'results/' + str(config.data_flag) + '_' + str(flag) + '/reg_' + str(model_name) + '.joblib')
