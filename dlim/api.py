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
    Compute the distance/constraints loss for a given model.

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
        dist = ((mat[:, newaxis, :] - mat[newaxis, :, :])**2).mean(dim=-1)
        losses += [(dist * const[pi]).mean()]
    return wei*sum(losses)/len(losses)



class DLIM_API():
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
            lr: float = 1e-3, 
            weight_decay: float = 1e-4, 
            nb_epoch: int = 100, 
            batch_size: int = 32, 
            emb_regularization: float = 0.0, 
            similarity_type: str = 'pearson', 
            temperature: float = 0.5, 
            save_path: Optional[str] = None) -> List[float]:
        """
        Train the model on the specified dataset.

        Args:
            data (torch.utils.data.Dataset): The dataset used for training.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay (L2 regularization) for the optimizer. Defaults to 1e-4.
            nb_epoch (int, optional): Number of epochs to train the model. Defaults to 100.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            emb_regularization (float, optional): Regularization factor for embedding layers. Defaults to 0.0.
            similarity_type (str, optional): Similarity measure to use in the spectral initialization. Defaults to 'pearson'. You can choose among
            ['pearson', 'cosine', 'euclidean']
            temperature (float, optional): Temperature scaling for similarity computation. Defaults to 0.5.
            save_path (str, optional): Path to save the trained model. If None, the model will not be saved. Defaults to None.

        Returns:
            List[float]: A list containing mean batch losses over epochs.
        """
        if self.flag_spectral :
            self.model.spec_init_emb(data, sim=similarity_type, temp=temperature, force=self.flag_spectral)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_f = GaussianNLLLoss()
        losses = []
        self.model.train()
        for _ in range(nb_epoch):
            loss_b, loss_l = [], []
            loader = DataLoader(data, batch_size=batch_size, shuffle=True)
            for bi, batch in enumerate(loader):
                optimizer.zero_grad()
                pfit, var, lat = self.model(batch[:, :-1].long())
                loss_mse = loss_f(pfit, batch[:, [-1]], var)
                if emb_regularization > 0:
                    loss_dist = (sum(torch.norm(el, p=2) for el in self.model.genes_emb)/len(self.model.genes_emb))**2
                    loss = loss_mse + emb_regularization * loss_dist
                else:
                    loss = loss_mse + emb_regularization
                loss.backward()
                optimizer.step()
                loss_b += [loss_mse.item()]
            losses += [mean(loss_b)]

        if save_path:
            try:
                torch.save(self.model, save_path)
                print(f"Model saved to {save_path}")
            except Exception as e:
                print(f"Failed to save model: {e}")

        return losses

    
    def predict(self, data: torch.Tensor, detach:  bool = True):
        """
        Make predictions using the trained model.

        Args:
            data (torch.Tensor): The input data to make predictions on.
            detach (bool, optional): If True, the result will be detached from the computation graph and converted to NumPy arrays. Defaults to True.

        Returns:
            Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
                - fit: The model predictions.
                - variance: The predicted variance.
                - lat: Latent variables (if applicable).
        """
        self.model.eval()

        # make sure that all data are on the same deviece
        device = next(self.model.parameters()).device
        data = data.to(device).long()

        fit, variance, lat = self.model(data)
    
        if detach:
            return fit.detach().cpu().numpy(), variance.detach().cpu().numpy(), lat.detach().cpu().numpy()
        else:
            return fit, variance, lat


    def plot(self, ax, data: Optional[Data_model] = None, fontsize: int =12, cols: list = [0, 1]):
        "only for pairs"
        min_x, max_x = self.model.genes_emb[cols[0]].min().item(), self.model.genes_emb[cols[0]].max().item()
        delta_x = 0.1*(max_x - min_x)
        min_y, max_y = self.model.genes_emb[cols[1]].min().item(), self.model.genes_emb[cols[1]].max().item()
        delta_y = 0.1*(max_y - min_y)
        x_v = np.linspace(min_x - delta_x, max_x + delta_x, 300)
        y_v = np.linspace(min_y - delta_y, max_y + delta_y, 300)
        x_m, y_m = np.meshgrid(x_v, y_v)
        data_np = np.concatenate((x_m[newaxis, :, :], y_m[newaxis, :, :]), axis=0)
        data_m = tensor(data_np).transpose(0, 2).reshape(-1, 2).to(tfloat)
        pred_l = self.model.predictor(data_m)[:, [0]].detach().numpy().reshape(300, 300).T


        norm = TwoSlopeNorm(vmin=min(-1e-6, pred_l.min()), vcenter=0, vmax=max(pred_l.max(), 1e-6,))
        ax.contourf(x_m, y_m, pred_l, cmap="bwr", alpha=0.4, norm=norm)

        ax.set_xlabel("$\\varphi_A$", fontsize=fontsize)
        ax.set_ylabel("$\\varphi_B$", fontsize=fontsize)

        if data is not None:
            _, _, lat = self.predict(data.data[:, :-1], detach=True)
            ax.scatter(lat[:, 0], lat[:, 1], c=data.data[:, -1], s=2, cmap="bwr", marker="x", norm=norm)
