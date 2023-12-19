# this file get to know how the regulization on latent space will change 
import sys 
sys.path.append('../')
from src.model import DLIM
from src.utils import Data_model, train
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from numpy import mean, logspace, std
from numpy.random import choice, seed
import matplotlib.pyplot as plt
from multiprocessing import Pool
import joblib 

data = Data_model("../data/data_env_1.csv", 2, const_file="../data/data_const.dat")
data_no = Data_model("../data/data_env_1.csv", 2)
val_frac = logspace(-2, 0.1, num=7)
val_id = choice(range(data.data.shape[0]), int(data.data.shape[0]*0.3))
train_full_id = [i for i in range(data.data.shape[0]) if i not in val_id]
train_full_data = data[train_full_id, :]
val_data = data[val_id, :]

def run_one(args):
    i, frac, wei_const = args
    seed(42 + i)
    train_id = choice(range(train_full_data.shape[0]), int(train_full_data.shape[0]*frac))
    model = DLIM(2, nb_state=37, hid=32, nb_layer=1)
    model_no = DLIM(2, nb_state=37, hid=32, nb_layer=1)

    train_data = train_full_data[train_id, :]

    _ = train(model, train_data, const=data.const, lr=1e-2, wei_const=wei_const, wei_dec=1e-3, nb_epoch=300, bsize=64)
    _ = train(model_no, train_data, lr=1e-2, wei_dec=1e-3, nb_epoch=300, bsize=64)

    fit = model(val_data[:, :-1].long())[0].detach().squeeze(-1)
    fit_no = model_no(val_data[:, :-1].long())[0].detach().squeeze(-1)
    cor_w = pearsonr(fit, val_data[:, -1])[0]
    cor_n = pearsonr(fit_no, val_data[:, -1])[0]
    return cor_w, cor_n

result = {}
res_w, res_n = [], []
pool = Pool(40)
for weight in [1, 5, 10, 20, 30, 40]:
    result[weight]  = {}
    for frac in val_frac:
        res = pool.map(run_one, [(i, frac, weight) for i in range(30)])
        tmp_w, tmp_n = zip(*res)
        res_w += [tmp_w]
        res_n += [tmp_n]
        result[weight][frac] = {
            'reg': tmp_w,
            'no': tmp_n
        }
joblib.dump(result, 'results/regulization_res_run.joblib')

