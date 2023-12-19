import sys 
sys.path.append('../')
from src.model import DLIM
from src.utils import Data_model, train
from src.sim_data import Simulated
from numpy import mean, var
from numpy.random import choice
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid
import numpy as np
from numpy.random import seed 
from multiprocessing import Pool
import joblib 

def run_one(args):
    i, cor = args
    seed(42 + i)
    tmp = []
    model = DLIM(2, nb_state=36, hid=16, nb_layer=0)
    data = Simulated(36, cor)
    train_id = choice(range(data.data.shape[0]), int(data.data.shape[0]*0.5))

    train_data = data[train_id, :]
    val_data = data[[i for i in range(data.data.shape[0]) if i not in train_data], :]

    # losses = train(model, train_data, nb_epoch=1000)
    losses = train(model, train_data, lr=1e-2, nb_epoch=400, bsize=64, wei_dec=1e-3)
    fit, var, lat = model(data[:, :-1].long(), detach=True)
    differences = lat[:, np.newaxis, :] - lat[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)
    tmp += [distances.flatten()]
    return tmp 


res_w, res_c = [], []
pool = Pool(20)
result = {}
for id, flag in enumerate(['add','comp','exp']):
    tmp_w = pool.map(run_one, [(i, flag) for i in range(20)])
    result[flag] = tmp_w
joblib.dump(result, 'results/simulated_data.joblib')


import matplotlib.pyplot as plt
import joblib 
import numpy as np 
result = joblib.load( 'results/simulated_data.joblib')
fig, bx = plt.subplots(1, 2, figsize=(2.5*2, 2.5))
for id, flag in enumerate(['comp', 'exp']):
    tmp1 = np.concatenate(np.array(result[flag]), axis = 1) 
    tmp2 = np.concatenate(np.array(result['add']), axis = 1) 
    tmp = list(tmp2) + list(tmp1)
    if flag == 'comp':
        lab = "$X+Y - (X \\times Y)$"
    elif flag == 'quad':
        lab = "$X \\times Y.$"
    elif flag == 'hat':
        lab = "$sin(X^2 + Y^2)$"
    elif flag == 'saddle':
        lab = "$X^2 - Y^2$"
    elif flag == 'exp':
        lab = "$ 10 \\times (e^{-(2-X)^2 - (2-Y)^2})$"
    for el in ["top", "right"]:
        bx[id].spines[el].set_visible(False)
    bx[id].hist(tmp, density=True, histtype="step", label=["$X+Y$", lab], bins=40)

    bx[id].legend(frameon=False, fontsize=8,  loc = 'upper right')
    bx[id].set_xlabel("Pair distance", fontsize=10)
    bx[id].set_ylabel("Freq.", fontsize=10)
plt.tight_layout()
plt.savefig("../fig/S6.png", dpi=300, transparent=True)
plt.show()



