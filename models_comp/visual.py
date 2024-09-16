import seaborn as sns
import matplotlib.pyplot as plt 
import sys 
sys.path.append('../')
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from numpy import mean, logspace, std
from numpy.random import choice, seed
import matplotlib.pyplot as plt
from multiprocessing import Pool
from dlim.utils import Data_model
# get result 
import joblib
import numpy as np 
import pandas as pd 
fig, ax = plt.subplot_mosaic([['bottom', 'bottom'],['left', 'right']],
                              constrained_layout=True, figsize = (5,5))
data = Data_model("../data/data_env_1.csv", 2)
color1 = '#e78ac3'  # pink
color2 = '#ffa500'  # modern orange
color3 = '#607d8b'  # light gray-blue
color4 = '#8da0cb' # 
colors = [color1, color2, color3, color4]
colors += ['#41afaa', 'k']
all_result = {}
data_dict = {
    'model': [],
    'acc': [],
    'data_name': []
    }
for id_data, data_name in enumerate(['fitness', 'epistasis']):
    if data_name == 'fitness':
        plot_f = 'left'
        
    else:
        plot_f = 'right'
    val_frac = logspace(-1, 0.1, num=7)

    for id, model in enumerate(['D-LIM', 'LR', 'ALM', 'LANTERN', 'MAVE-NN']):
       
        result_nn_reg = joblib.load('results/' + str(data_name) + '/reg_' + str(model) + '.joblib')
        
        fracs = list(result_nn_reg.keys())
        res_c = [result_nn_reg[frac] for frac in fracs] 
        res_w = result_nn_reg[fracs[-1]]
        data_dict['acc'] += [float(el) for el in res_c[-1] ]
        data_dict['model'] += [str(model) for el in res_w]
        data_dict['data_name'] += [str(data_name) for el in res_w]
        if plot_f == 'right':
            ax[plot_f].plot(val_frac* data.data.shape[0], [mean(el) for el in res_c], c=colors[id], lw=2, label=model)
            ax[plot_f].scatter(val_frac* data.data.shape[0], [mean(el) for el in res_c], c=colors[id], s=15)
            ax[plot_f].errorbar(val_frac* data.data.shape[0], [mean(el) for el in res_c], yerr=[np.quantile(el, 0.95) - mean(el) for el in res_c], c=colors[id])
        else:
            ax[plot_f].plot(val_frac* data.data.shape[0], [mean(el) for el in res_c], c=colors[id], lw=2)
            ax[plot_f].scatter(val_frac* data.data.shape[0], [mean(el) for el in res_c], c=colors[id], s=15)
            ax[plot_f].errorbar(val_frac* data.data.shape[0], [mean(el) for el in res_c], yerr=[np.quantile(el, 0.95) - mean(el) for el in res_c], c=colors[id])
        print(model)
        print([mean(el) for el in res_c])
    ax[plot_f].set_xscale("log")
    for el in ["top", "right"]:
        ax[plot_f].spines[el].set_visible(False)
    ax[plot_f].set_ylabel(f"$\\rho$")
    ax[plot_f].set_xlabel("nb. data points")
    ax[plot_f].set_title(data_name)
    

lgd = fig.legend(frameon=False, ncol = 5, loc='lower center',bbox_to_anchor=(0.5, -0.05, 0, 0),)
df = pd.DataFrame.from_dict(data_dict)
axe = sns.boxplot(data=df, x="model", y="acc", hue="data_name",  palette=["C0", "C1"], ax=ax['bottom'])
axe.set_title('cross validation on fitness and epistasis')
# Remove the top and right spines
axe.spines["top"].set_visible(False)
axe.spines["right"].set_visible(False)
# statistical annotation
axe.set_ylim([0.8, 1.0])
axe.set_xlabel("Models", fontsize = 10)
axe.set_ylabel(f"$\\rho$", fontsize = 10)
fig.tight_layout()
fig.savefig('../img/S5b_model_comp.svg', dpi = 300, transparent = True, bbox_extra_artists=(lgd,), bbox_inches='tight')
fig.show()
