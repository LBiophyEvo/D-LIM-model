from torch import tensor 
from torch import tensor, float32 as tfloat, cat as tcat
import matplotlib.pyplot as plt 
from numpy import mean, linspace, meshgrid
import numpy as np 

def plot_figure(model, id_gene, save_name = '../fig/S_simulate.png' , data = None):
    fig, ax = plt.subplots(1, 3, figsize=(3*2.5, 2.5))
    for flag in range(3):
        if flag == 0:
            min_x, max_x = model.genes[0].min().item(), model.genes[0].max().item()
            delta_x = 0.1*(max_x - min_x)
            min_y, max_y = model.genes[1].min().item(), model.genes[1].max().item()
            delta_y = 0.1*(max_y - min_y)
            x_v = linspace(min_x - delta_x, max_x + delta_x, 200)
            y_v = linspace(min_y - delta_y, max_y + delta_y, 200)
            x_m, y_m = meshgrid(x_v, y_v)
            z0 = model.genes[2][id_gene].item()
            z_m = y_m*0 + z0
            data_np = np.concatenate((x_m[np.newaxis, :, :], y_m[np.newaxis, :, :], z_m[np.newaxis, :, :]), axis=0)
            data_m = tensor(data_np).transpose(0, 2).reshape(-1, 3).to(tfloat)
            pred_l = model.epi(data_m)[:, [0]].detach().numpy().reshape(200, 200).T
            vmin, vmax = pred_l.min() , pred_l.max() 
            ax[flag].contourf(x_m, y_m, pred_l, cmap="bwr", alpha=0.4, vmin = vmin, vmax = vmax)
            ax[flag].set_xlabel("$Z^1$")
            ax[flag].set_ylabel("$Z^2$")


            
            if data is not None:
                data_f = data[data[:,2] == id_gene]
                try:
                    fit, var, lat = model.forward(data_f[:, :-1].int(), detach=True)
                except:
                    fit, var, lat = model.forward(data_f[:, :-1].long(), detach=True)
    
                ax[flag].scatter(lat[:, 0], lat[:, 1], c=data_f[:, -1], s=2, cmap="bwr", marker="x", vmin = vmin, vmax = vmax)
            ax[flag].set_title(f"$Z^3$")
            
        elif flag == 1:
            min_z, max_z = model.genes[2].min().item(), model.genes[2].max().item()
            delta_z = 0.1*(max_z - min_z)
            min_y, max_y = model.genes[1].min().item(), model.genes[1].max().item()
            delta_y = 0.1*(max_y - min_y)
            z_v = linspace(min_z - delta_z, max_z + delta_z, 200)
            y_v = linspace(min_y - delta_y, max_y + delta_y, 200)
            z_m, y_m = np.meshgrid(z_v, y_v)
            x0 = model.genes[0][id_gene].item()
            x_m = z_m*0 + x0 
            data_np = np.concatenate((x_m[np.newaxis, :, :], y_m[np.newaxis, :, :], z_m[np.newaxis, :, :]), axis=0)
            data_m = tensor(data_np).transpose(0, 2).reshape(-1, 3).to(tfloat)
            pred_l = model.epi(data_m)[:, [0]].detach().numpy().reshape(200, 200).T
            vmin, vmax = pred_l.min(), pred_l.max()
            ax[flag].contourf(y_m, z_m, pred_l, cmap="bwr", alpha=0.4)
            ax[flag].set_xlabel("$Z^2$")
            ax[flag].set_ylabel("$Z^3$")

            if data is not None:
                data_f = data[data[:,0] == id_gene]
                try:
                    fit, var, lat = model.forward(data_f[:, :-1].int(), detach=True)
                except:
                    fit, var, lat = model.forward(data_f[:, :-1].long(), detach=True)
    
                ax[flag].scatter(lat[:, 1], lat[:, 2], c=data_f[:, -1], s=2, cmap="bwr", marker="x")
            ax[flag].set_title(f"$Z^1$")
        elif flag == 2:
            min_z, max_z = model.genes[2].min().item(), model.genes[2].max().item()
            delta_z = 0.1*(max_z - min_z)
            min_x, max_x = model.genes[0].min().item(), model.genes[0].max().item()
            delta_x = 0.1*(max_x - min_x)
            z_v = linspace(min_z - delta_z, max_z + delta_z, 200)
            x_v = linspace(min_x - delta_x, max_x + delta_x, 200)
            z_m, x_m = np.meshgrid(z_v, x_v)
            y0 = model.genes[1][id_gene].item()
            y_m = z_m*0 + x0 
            data_np = np.concatenate((x_m[np.newaxis, :, :], y_m[np.newaxis, :, :], z_m[np.newaxis, :, :]), axis=0)
            data_m = tensor(data_np).transpose(0, 2).reshape(-1, 3).to(tfloat)
            pred_l = model.epi(data_m)[:, [0]].detach().numpy().reshape(200, 200).T
            ax[flag].contourf(x_m, z_m, pred_l, cmap="bwr", alpha=0.4)
            ax[flag].set_xlabel("$Z^1$")
            ax[flag].set_ylabel("$Z^3$")

            if data is not None:
                data_f = data[data[:,1] == id_gene]
                try:
                    fit, var, lat = model.forward(data_f[:, :-1].int(), detach=True)
                except:
                    fit, var, lat = model.forward(data_f[:, :-1].long(), detach=True)
                ax[flag].scatter(lat[:, 0], lat[:, 2], c=data_f[:, -1], s=2, cmap="bwr", marker="x")
            ax[flag].set_title(f"$Z^2$")
    fig.tight_layout()
    fig.savefig(save_name, dpi = 300, transparent=True)