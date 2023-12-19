# this scirpt is used to test different parameters and their influences on results 
import sys 
sys.path.append('../')
from src.model import DLIM
from src.utils import Data_model, train
from numpy import mean, linspace, logspace, sqrt 
from numpy.random import choice
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from numpy.random import choice, seed
import joblib 
from multiprocessing import Pool
import argparse 

class algo_run_one():
    def __init__(self, train_full_data, val_data):
        self.train_full_data = train_full_data
        self.val_data  = val_data 
    def run_one(self, arg):
        nb_epoch = 500
        i, frac = arg[2], arg[3]
        seed(42 + i)
        model = DLIM(2, nb_state=50, hid=arg[0], nb_layer=arg[1])
        train_id = choice(range(self.train_full_data.shape[0]), int(self.train_full_data.shape[0]*frac))
        train_data = self.train_full_data[train_id, :]

        losses = train(model, train_data, lr=1e-3, nb_epoch=nb_epoch, bsize=16, val_data=val_data, wei_dec=1e-3)
        model.eval()
        fit, var, _ = model(self.val_data[:, :-1].long(), detach=True)
        pear = pearsonr(fit.flatten(), self.val_data[:, [-1]].flatten())[0]
        spearman = spearmanr(fit.flatten(), self.val_data[:, [-1]].flatten())[0]
        r2 = r2_score(fit.flatten(), self.val_data[:, [-1]].flatten())
        mse = (((fit.flatten() - self.val_data[:, [-1]].flatten())**2).mean()).item()
        return [pear, spearman, r2, mse]


parser = argparse.ArgumentParser(description='Process different layers.') 
data = Data_model("../data/data_env_1.csv", 2)
val_frac = logspace(-1, 0.1, num=7)
val_id = choice(range(data.data.shape[0]), int(data.data.shape[0]*0.3))
train_full_id = [i for i in range(data.data.shape[0]) if i not in val_id]
train_full_data = data[train_full_id, :]
val_data = data[val_id, :]

pool = Pool(20)
result = {}

neuros_l = [2, 4, 8, 16, 32]
layers_l = [0, 1, 2, 3]

# neuros_l = [2]
# layers_l = [0]
run_iter = 10
model_run_one = algo_run_one(train_full_data, data) 
parser.add_argument('-l', "--layer", default= 0, type = int)
args = vars(parser.parse_args())
layer = args['layer']
print(layer)
for neuro in neuros_l:
    # for layer in layers_l:
    for frac in val_frac:  
        tmp_w = pool.map(model_run_one.run_one, [(neuro, args['layer'], i, frac) for i in range(run_iter)])
        result[(frac, neuro, layer)] = tmp_w
joblib.dump(result, 'results/si_1_parameters_'+ str(args['layer']) + '.joblib')
