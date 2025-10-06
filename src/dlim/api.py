from dlim.model import DLIM
from dlim.dataset import Data_model
from numpy import mean, newaxis
from torch import tensor, optim, rand, zeros, no_grad
from matplotlib.colors import TwoSlopeNorm
from torch import float32 as tfloat
from torch.nn import GaussianNLLLoss
from torch.utils.data import Dataset, DataLoader
import torch 
from typing import Optional, Tuple, List
import numpy as np 

def dist_loss(lat, const, wei=1):
    """
    Compute the distance/constraints loss for a given model (mean square error).

    Args:
        lat (tensor): The latent representation of the data.
        const (dict): Dictionary of constraints.
        wei (float, optional): Weight of the constraint in the loss. Defaults to 1.

    Returns:
        float: The computed distance/constraints loss.
    """
    losses = []
    for pi in const:
        mat = lat[pi]
        # Compute pairwise squared distances and apply constraint mask
        dist = ((mat[:, newaxis, :] - mat[newaxis, :, :])**2).mean(dim=-1)
        losses += [(dist * const[pi]).mean()]
    return wei*sum(losses)/len(losses)

class DLIM_API():
    """
    API class for training, evaluating, and visualizing the DLIM model.

    Attributes:
        flag_spectral (bool): Whether to use spectral initialization.
        model (DLIM): The DLIM model instance.

    Args:
        model (DLIM): DLIM model to use.
        flag_spectral (bool, optional): Use spectral initialization. Defaults to False.
        load_model (str, optional): Path to load a pre-trained model. Defaults to None.
    """
    def __init__(self, model: DLIM, flag_spectral: bool = False, load_model: Optional[str] = None):
        self.flag_spectral = flag_spectral 

        # Load model if load_model path is provided
        if load_model is not None:
            try:
                self.model = torch.load(load_model)
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {load_model}. Error: {e}")
        else:
            self.model = model 
    
    def fit(self, 
            data: Data_model, 
            test_data: Optional[Data_model] = None,
            lr: float = 1e-3, 
            weight_decay: float = 1e-4, 
            nb_epoch: int = 100, 
            batch_size: int = 32, 
            emb_regularization: float = 0.0, 
            similarity_type: str = 'pearson', 
            temperature: float = 0.5, 
            max_patience: int = 10, 
            save_path: Optional[str] = None,
            return_best_model: bool = False) -> List[float]:
        """
        Train the DLIM model on the specified dataset.

        Args:
            data (Data_model): The dataset used for training.
            test_data (Data_model, optional): Dataset for validation/testing.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay (L2 regularization) for the optimizer. Defaults to 1e-4.
            nb_epoch (int, optional): Number of epochs to train the model. Defaults to 100.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            emb_regularization (float, optional): Regularization factor for embedding layers. Defaults to 0.0.
            similarity_type (str, optional): Similarity measure for spectral initialization. Defaults to 'pearson'.
            temperature (float, optional): Temperature scaling for similarity computation. Defaults to 0.5.
            max_patience (int, optional): Early stopping patience. Defaults to 10.
            save_path (str, optional): Path to save the trained model. If None, the model will not be saved.
            return_best_model (bool, optional): Whether to return the best model and test loss.

        Returns:
            List[float] or Tuple: Training and optionally test losses, best model, and best test loss.
        """
        if self.flag_spectral :
            # Spectral initialization of embeddings
            self.model.spec_init_emb(data, sim=similarity_type, temp=temperature, force=self.flag_spectral)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler_step = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10
        )
        loss_f = GaussianNLLLoss()
        loss_mse_f = torch.nn.MSELoss()
        losses = []
        self.model.train()
        loss_best_test = np.inf
        patience = 0 
        for _ in range(nb_epoch):
            loss_b, loss_l = [], []
            loader = DataLoader(data, batch_size=batch_size, shuffle=True)
            for bi, batch in enumerate(loader):
                optimizer.zero_grad()
                # Forward pass
                pfit, var, lat = self.model(batch[:, :-1].long())
                # Compute Gaussian NLL and MSE losses
                loss_gaussian = loss_f(pfit, batch[:, [-1]], var)
                loss_mse = loss_mse_f(pfit, batch[:, [-1]])
                # Embedding regularization
                if emb_regularization > 0:
                    loss_dist = (sum(torch.norm(el, p=2) for el in self.model.genes_emb)/len(self.model.genes_emb))**2
                    loss = loss_gaussian + emb_regularization * loss_dist
                else:
                    loss = loss_gaussian + emb_regularization
                
                loss.backward()
                optimizer.step()
                loss_b += [loss_mse.item()]
            if test_data != None:
                # Evaluate on test data
                loss_test = []
                loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
                with torch.no_grad():
                    self.model.eval()
                    for bi, batch in enumerate(loader):
                        pfit, var, lat = self.model(batch[:, :-1].long())
                        loss_mse = loss_mse_f(pfit, batch[:, [-1]])
                        loss_test.append(loss_mse.item())
                loss_test_epoch = np.mean(loss_test)
                losses += [(mean(loss_b), loss_test_epoch)]

                # Track best model based on test loss
                if loss_best_test > loss_test_epoch:
                    loss_best_test = loss_test_epoch.copy()
                    best_model = self.model 
                    patience = 0 
                else:
                    patience += 1 
            else:
                best_model = self.model 
                loss_best_test = None 
                losses += [mean(loss_b)]
            # Adapt the learning rate 
            if patience > max_patience:
                scheduler_step.step()
                # break 
        
        # Save the best model in the save_path 
        self.model = best_model      
        if save_path:
            try:
                torch.save(best_model, save_path)
                print(f"Model saved to {save_path}")
            except Exception as e:
                print(f"Failed to save model: {e}")
        if return_best_model:
            return losses, best_model, loss_best_test
        else:
            return losses

    def predict(self, data: torch.Tensor, detach:  bool = True):
        """
        Make predictions using the trained model.

        Args:
            data (torch.Tensor): The input data to make predictions on.
            detach (bool, optional): If True, the result will be detached from the computation graph and converted to NumPy arrays. Defaults to True.

        Returns:
            Tuple: fit (predictions), variance, and latent variables.
        """
        self.model.eval()

        # Ensure data is on the same device as the model
        device = next(self.model.parameters()).device
        data = data.to(device).long()

        fit, variance, lat = self.model(data)
    
        if detach:
            return fit.detach().cpu().numpy(), variance.detach().cpu().numpy(), lat.detach().cpu().numpy()
        else:
            return fit, variance, lat

    def plot(self, ax, data: Optional[Data_model] = None, fontsize: int =12, cols: list = [0, 1], xy_labels: Optional[List] = None):
        """
        Visualize the learned landscape and optionally the data points.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes to plot on.
            data (Data_model, optional): Dataset to plot points from.
            fontsize (int, optional): Font size for axis labels.
            cols (list, optional): Indices of latent dimensions to plot.
            xy_labels (list, optional): Custom axis labels.
        """
        # Compute grid for contour plot
        min_x, max_x = self.model.genes_emb[cols[0]].min().item(), self.model.genes_emb[cols[0]].max().item()
        delta_x = 0.1*(max_x - min_x)
        min_y, max_y = self.model.genes_emb[cols[1]].min().item(), self.model.genes_emb[cols[1]].max().item()
        delta_y = 0.1*(max_y - min_y)
        x_v = np.linspace(min_x - delta_x, max_x + delta_x, 300)
        y_v = np.linspace(min_y - delta_y, max_y + delta_y, 300)
        # Get meshgrid 
        x_m, y_m = np.meshgrid(x_v, y_v)
        data_np = np.concatenate((x_m[newaxis, :, :], y_m[newaxis, :, :]), axis=0)
        data_m = tensor(data_np).transpose(0, 2).reshape(-1, 2).to(tfloat)
        # Predict the fitness value for the point on meshgrid 
        pred_l = self.model.predictor(data_m)[:, [0]].detach().numpy().reshape(300, 300).T

        # Plot contour
        norm = None
        ax.contourf(x_m, y_m, pred_l, cmap="bwr", alpha=0.4, norm=norm)
        if xy_labels != None:
            ax.set_xlabel(xy_labels[0], fontsize=fontsize)
            ax.set_ylabel(xy_labels[1], fontsize=fontsize)
        else:
            ax.set_xlabel("$\\varphi_1$", fontsize=fontsize)
            ax.set_ylabel("$\\varphi_2$", fontsize=fontsize)

        # Optionally plot data points
        if data is not None:
            _, _, lat = self.predict(data.data[:, :-1], detach=True)
            ax.scatter(lat[:, 0], lat[:, 1], c=data.data[:, -1], s=2, cmap="bwr", marker="x", norm=norm)