import sys 
sys.path.append('../')
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

from joblib import Parallel, delayed
from time import time
import joblib 
import argparse 
from tqdm import tqdm 

def read_data(file, nb_var = 2, sep = ','):
    data = pd.read_csv(file, header=None, sep = sep)
    data = data.dropna()
    fit = data.iloc[:,nb_var]
    comb = data.iloc[:,0].astype(str) + '_gene1' + ':' +  data.iloc[:,1].astype(str)+ '_gene2'
    df = pd.DataFrame({'substitutions': list(comb), 'phenotype': list(fit)})
    return df

class parel_run():
    def __init__(self, df, train_full_id, Xtest, ytest):
        self.df = df
        self.train_full_id = train_full_id
        self.Xtest = Xtest
        self.ytest = ytest  
    def run_one_lantern(self, i, frac):
        seed(42 + i)
        len_train = len(self.train_full_id)
        train_id = choice(self.train_full_id, int(len_train*frac))
        left_id = [i for i in self.train_full_id if i not in train_id]
        df_copy = self.df.copy()
        df_copy['phenotype'].iloc[left_id] = np.NaN
        ds = Dataset(df_copy)
        K = 4
        basis = VariationalBasis.fromDataset(ds, K=K)
        surface = Phenotype.fromDataset(ds, K=K, Ni=200, inducScale=1.0)
        model = Model(basis, surface, GaussianLikelihood())

        loss = model.loss(N=len(ds))
        Xtrain, ytrain = ds[train_id]
    
        E = 1000
        optimizer = Adam(loss.parameters(), lr=0.01)
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
            Z_t = model.basis(self.Xtest)
        yhat = model.surface(Z_t)
        ypredict = (yhat.mean).detach().numpy() 
        pear = pearsonr(self.ytest.detach().numpy().flatten(), ypredict)[0]

        return pear


if __name__ == '__main__':
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
    df = read_data(file = data_path, sep = sep)
    

    val_frac = logspace(-1, 0.1, num=7)
    seed(42)
    val_id = choice(range(df.shape[0]), int(df.shape[0]*0.3))
    train_full_id = [i for i in range(df.shape[0]) if i not in val_id]
    ds_copy = Dataset(df)
    Xtest, ytest = ds_copy[val_id]
    df['phenotype'].iloc[val_id] = np.NaN

    run_iter = parel_run(df, train_full_id, Xtest, ytest)

    start_time = time()
    result = {}
    print(model_name)
    for el, frac in enumerate(val_frac):
        print(frac)
        twd = []
        for i_time in tqdm(range(30)):
            r = run_iter.run_one_lantern(i_time, frac) 
            twd.append(r)
        result[frac] = twd
    end_time = time()
    print('Lantern using time:', end_time - start_time)
    joblib.dump(result, 'results/' + str(config.data_flag) + '_' + str(flag) + '/reg_' + str(model_name) + '.joblib')
