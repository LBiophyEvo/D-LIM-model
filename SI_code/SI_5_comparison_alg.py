import sys 
sys.path.append('../')
from src.model import DLIM, Regression, Add_Latent
from src.utils import Data_model, train, train_reg
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from numpy import mean, logspace, std
from numpy.random import choice, seed
import matplotlib.pyplot as plt
from multiprocessing import Pool
from src.sim_data import Simulated

data = Data_model("../data/data_env_1.csv", 2)
val_frac = logspace(-1, 0.1, num=7)
val_id = choice(range(data.data.shape[0]), int(data.data.shape[0]*0.3))
train_full_id = [i for i in range(data.data.shape[0]) if i not in val_id]
train_full_data = data[train_full_id, :]
val_data = data[val_id, :]

def run_one(args):
    i, frac = args
    seed(42 + i)
    train_id = choice(range(train_full_data.shape[0]), int(train_full_data.shape[0]*frac))
    model = DLIM(2, nb_state=50, hid=31, nb_layer=1)
    model_reg = Regression(2, nb_state=37)
    model_add_lantern = Add_Latent(2, nb_state=50, hid=32, nb_layer=1)

    train_data = train_full_data[train_id, :]

    _ = train(model, train_data, lr=1e-2, wei_dec=1e-3, nb_epoch=300, bsize=64)
    _ = train_reg(model_reg, train_data, lr=1e-2, nb_epoch=300, bsize=64)
    _ = train(model_add_lantern, train_data, lr=1e-2, wei_dec=1e-3, nb_epoch=300, bsize=64)

    fit = model(val_data[:, :-1].long())[0].detach().squeeze(-1)
    fit_reg = model_reg(val_data[:, :-1].long()).detach().squeeze(-1)
    fit_add_latern = model_add_lantern(val_data[:, :-1].long())[0].detach().squeeze(-1)
    cor_w = pearsonr(fit, val_data[:, -1])[0]
    cor_r = pearsonr(fit_reg, val_data[:, -1])[0]
    cor_l = pearsonr(fit_add_latern, val_data[:, -1])[0]
    return cor_w, cor_r, cor_l

import joblib 
res_w, res_c, res_l= [], [], []
pool = Pool(20)
result = {}
for frac in val_frac:
    tmp_w = pool.map(run_one, [(i, frac) for i in range(10)])
    res_w += [[w for w, _, _ in tmp_w]]
    res_c += [[c for _, c, _ in tmp_w]]
    res_l += [[l for _, _, l in tmp_w]]
    result[frac] = {
        'nn':  [[w for w, _, _ in tmp_w]],
        'reg':[[c for _, c, _ in tmp_w]],
        'add_lan': [[l for _, _, l in tmp_w]] 
    }
joblib.dump(result, 'results/nn_reg.joblib')


## train lantern 

from csv import reader 
import pandas as pd 
from lantern.model.basis import VariationalBasis
from lantern.model.surface import Phenotype 

from lantern.model import Model
from lantern.model.likelihood import GaussianLikelihood
from torch.optim import Adam
from random import shuffle 
import numpy as np 
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import torch 
from lantern.dataset import Dataset
from scipy.stats import pearsonr
from numpy import logspace
from numpy.random import choice 
import matplotlib.pyplot as plt 
from multiprocessing import Pool
from numpy.random import choice, seed

def read_data(file, nb_var = 2):
    data = pd.read_csv(file, header=None)
    data = data.dropna()
    fit = data.iloc[:,nb_var]
    comb = data.iloc[:,0].astype(str) + '_gene1' + ':' +  data.iloc[:,1].astype(str)+ '_gene2'
    df = pd.DataFrame({'substitutions': list(comb), 'phenotype': list(fit)})
    return df

K = 2
df = read_data(file = "../data/data_env_1.csv")
ds = Dataset(df)
len(ds)

val_frac = logspace(-1, 0.1, num=7)
val_id = choice(range(df.shape[0]), int(df.shape[0]*0.3))
train_full_id = [i for i in range(df.shape[0]) if i not in val_id]
len_train = len(train_full_id)

def run_one(i, frac):
    seed(42 + i)
    train_id = choice(range(len_train), int(len_train*frac))

    basis = VariationalBasis.fromDataset(ds, K=K) 

    surface = Phenotype.fromDataset(ds, K=K, Ni=100, inducScale=0.8)
    model = Model(basis, surface, GaussianLikelihood())

    loss = model.loss(N=len(ds))
    Xtrain, ytrain = ds[train_id]
    Xtest,ytest = ds[val_id]

    E = 1000
    optimizer = Adam(loss.parameters(), lr=0.01, weight_decay=1e-5)
    hist = []
    halpha = np.zeros((E, K))

    for i in range(E):
        
        optimizer.zero_grad()
        yhat = model(Xtrain)
        lss = loss(yhat, ytrain)
        total = sum(lss.values())
        total.backward()
        optimizer.step()
        
        hist.append(total.item())
        halpha[i, :] = basis.qalpha(detach=True).mean.numpy()
        

    with torch.no_grad():
        Z_t = model.basis(Xtest)
    yhat = model.surface(Z_t)
    ypredict = (yhat.mean).detach().numpy() 
    pear = pearsonr(ytest.detach().numpy().flatten(), ypredict)[0]

    return pear



from joblib import Parallel, delayed
from time import time
import joblib 
start_time = time()
result = {}
for frac in val_frac:
    r = Parallel(n_jobs=40)(delayed(run_one)(i, frac) for i in range(20))
    result[frac] = r 
end_time = time()
print('using time:', end_time - start_time)
joblib.dump(result, 'results/lantern_result.joblib')



