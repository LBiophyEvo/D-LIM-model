
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

from multiprocessing import Pooldlim

class run_mevv():
    def __init__(self, train_full_id, L, alphabet, data_df, val_data, val_id):
        self.train_full_id = train_full_id
        self.L = L 
        self.alphabet = alphabet 
        self.data_df = data_df
        self.val_data = val_data
        self.val_id = val_id 
    def run_one(self, args):
        i, frac = args
        # seed(42 + i)
        train_id = choice(self.train_full_id, int(len(self.train_full_id)*frac))

        # Define model
        model_mave = mavenn.Model(L=self.L,
                                alphabet=self.alphabet,
                                gpmap_type='additive',
                                regression_type='GE',
                                ge_noise_model_type='SkewedT',
                                ge_heteroskedasticity_order=2)

        
        # Set training data
        model_mave.set_data(x=self.data_df['x'][train_id],
                            y=self.data_df['y'][train_id])

        # Train model
        model_mave.fit(learning_rate=1e-3,
                        epochs=1000,
                        batch_size=64,
                        early_stopping=True,
                        early_stopping_patience=25,
                        verbose=False)

    

    
        fit_mave = model_mave.x_to_yhat(self.data_df["x"][self.val_id])
        cor_m = pearsonr(fit_mave, self.val_data[:, -1])[0]
        return cor_m



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train all models')
    parser.add_argument('--flag', type=str, default='fitness')
    config = parser.parse_args()
    flag =  config.flag
    print(flag)
    model_name = 'MAVE-NN'
    if flag == 'fitness':
        data_path = "../data/data_env_1.csv"
    elif flag == 'epistasis':
        data_path = "../data/data_epis_1.csv"

    # seed(42)
    data = Data_model(data_path, 2)
    # it's data_epis_1.csv
    data_df = pd.read_csv("../data/kemble_"+ str(flag) +".dat", sep="\t")
    L = len(data_df.loc[0,'x'])
    alphabet = ["A", "C", "G", "U", "T", "N"]
    val_frac = logspace(-1, 0.1, num=7)
    val_id = choice(range(data.data.shape[0]), int(data.data.shape[0]*0.3))
    train_full_id = [i for i in range(data.data.shape[0]) if i not in val_id]
    train_full_data = data[train_full_id, :]
    val_data = data[val_id, :]

    run_f = run_mevv(train_full_id, L, alphabet, data_df, val_data, val_id)
    pool = Pool(30)
    frac = 1.0
    tmp_w = pool.map(run_f.run_one, [(i, frac) for i in range(30)])
    print(mean(tmp_w))
    # max_iter = 30 
    # print(model_name)
    # result = {}
    # for frac in val_frac:
    #     print(frac)
    #     tmp_w = []
    #     for i in range(max_iter):
    #         r = run_f.run_one(i, frac) 
    #         tmp_w.append(r)
    #     result[frac] = tmp_w
    # joblib.dump(result, 'results_v2/' + str(flag) + '/reg_' + str(model_name) + '.joblib')