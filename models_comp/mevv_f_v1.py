
import sys 
sys.path.append('../')
from src.utils import Data_model, train, train_reg
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from numpy import mean, logspace, std
from numpy.random import choice, seed
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd
import mavenn
import argparse
import joblib 

from multiprocessing import Pool

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
print(flag)
model_name = 'MAVE-NN'

data = Data_model(data_path, 2, sep = sep)
# it's data_epis_1.csv
data_df = pd.read_csv("../data/kemble_"+ str(flag) +".dat", sep="\t")
L = len(data_df.loc[0,'x'])
alphabet = ["A", "C", "G", "U", "T", "N"]
val_frac = logspace(-1, 0.1, num=7)
seed(42)
val_id = choice(range(data.data.shape[0]), int(data.data.shape[0]*0.3))
train_full_id = [i for i in range(data.data.shape[0]) if i not in val_id]
train_full_data = data[train_full_id, :]
val_data = data[val_id, :]

def run_one(args):
    i, frac = args
    seed(42 + i)

    train_id = choice(train_full_id, int(len(train_full_id)*frac))
    # train_id = choice(range(train_full_data.shape[0]), int(train_full_data.shape[0]*frac))
    # Define model
    model_mave = mavenn.Model(L=L,
                              alphabet=alphabet,
                              gpmap_type='additive',
                              regression_type='GE',
                              ge_noise_model_type='SkewedT',
                              ge_heteroskedasticity_order=2)
    # Set training data
    model_mave.set_data(x=data_df['x'][train_id],
                        y=data_df['y'][train_id])

    # Train model
    model_mave.fit(learning_rate=1e-3,
                    epochs=1000,
                    batch_size=64,
                    early_stopping=True,
                    early_stopping_patience=25,
                    verbose=False)

    fit_mave = model_mave.x_to_yhat(data_df["x"][val_id])
    cor_m = pearsonr(fit_mave, val_data[:, -1])[0]
    return cor_m



# pool = Pool(30)
max_iter = 30 
print(model_name)
result = {}
for frac in val_frac:
    # print(frac)
    # tmp_w = pool.map(run_one, [(i, frac) for i in range(60)])
    # print(mean(tmp_w))
    tmp_w = []
    for i in range(max_iter):
        r = run_one(i, frac) 
        tmp_w.append(r)
        print(r)
    result[frac] = tmp_w
joblib.dump(result, 'results/' + str(config.data_flag) + '_' + str(flag) + '/reg_' + str(model_name) + '.joblib')
