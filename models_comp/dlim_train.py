import sys 
sys.path.append('../')
from dlim.model import DLIM 
from dlim.dataset import Data_model
from dlim.api import DLIM_API
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from numpy import mean
from numpy.random import choice
import pandas as pd
from tqdm import tqdm  
import matplotlib.pyplot as plt 
import numpy as np 
from multiprocessing import Pool
import pandas as pd
import argparse
from tqdm import tqdm 
import joblib 
from numpy.random import seed
import torch 
import os 

parser = argparse.ArgumentParser(description='Train all models')
parser.add_argument('--flag', type=str, default='fitness')
parser.add_argument('--data_flag', type=str, default='harry')

config = parser.parse_args()
flag =  config.flag
model_name = 'dlim'
if config.data_flag == 'harry':
    if config.flag == 'epistasis':
        nb_layer = 1
    else:
        nb_layer = 0 
    hparam = {
        'nb_state': 50,
        'emb': 1,
        'hid': 32,
        'nb_layer': nb_layer,
        'lr':1e-3, 
        'wei_dec':1e-3, 
        'nb_epoch':400, 
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
        'lr':1e-3, 
        'wei_dec':1e-3, 
        'nb_epoch':300, 
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
        'hid': 64,
        'nb_layer': 1,
        'lr':1e-3, 
        'wei_dec':1e-3, 
        'nb_epoch':300, 
        'bsize': 256,
        'sep': ','
    }
    data_path = "../data/protein_inter/tables1.csv"

df_data = pd.read_csv(data_path, sep = hparam['sep'], header = None)
data = Data_model(data=df_data, n_variables=2)


seed(42)
val_frac = np.logspace(-1, 0.1, num=7)
val_id = choice(range(data.data.shape[0]), int(data.data.shape[0]*0.3))
train_full_id = [i for i in range(data.data.shape[0]) if i not in val_id]
val_data = data.subset(val_id)
num_train_full_data = len(train_full_id)
def run_one(i, frac):
    seed(42 + i)
    train_id = choice(train_full_id, int(num_train_full_data*frac))
    train_data = data.subset(train_id)
    
    model = DLIM(n_variables = train_data.nb_val, hid_dim = hparam['hid'], nb_layer = hparam['nb_layer'])
    dlim_regressor = DLIM_API(model=model, flag_spectral=True)
    _ = dlim_regressor.fit(train_data, lr = hparam['lr'], nb_epoch=hparam['nb_epoch'], \
                                batch_size=hparam['bsize'], emb_regularization=0)

    fit, var, _  = dlim_regressor.predict(val_data.data[:,:-1], detach=True) 



    cor, pval = pearsonr(fit.flatten(), val_data.data[:, [-1]].flatten())

    return cor 

max_iter = 50
result = {}

print(model_name)
for frac in val_frac:
    print(frac)
    tmp_w = []
    for i in tqdm(range(max_iter)):
        r = run_one(i, frac) 
        tmp_w.append(r)
    result[frac] = tmp_w
path_save = 'results/' + str(config.data_flag) + '_' + str(flag) 
os.makedirs(path_save, exist_ok=True)
joblib.dump(result, path_save + '/reg_' + str(model_name) + '.joblib')
