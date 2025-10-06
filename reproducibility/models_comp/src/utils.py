from torch.utils.data import Dataset, DataLoader
from torch import tensor, optim, rand, zeros, no_grad
from torch.nn import GaussianNLLLoss
import torch
from pandas import read_csv
import numpy as np
from numpy import mean, newaxis, argsort, diag, zeros, intersect1d
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine, euclidean
from scipy.linalg import eigh


class Data_model(Dataset):
    """
    A custom dataset class for handling data in machine learning models.

    Attributes:
        all_mut (list): List of sets, each containing all mutations for a variable.
        mut_to_index (list): List of dictionaries mapping mutations to indices.
        data (tensor): Tensor representation of the data.
        const (dict or None): Constraints if provided, otherwise None.

    Args:
        infile (str): Path to the input file containing data.
        nb_var (int): Number of variables in the dataset.
        const_file (str, optional): Path to the constraints file. Defaults to None.
    """

    def __init__(self, infile, nb_var, sep=",", header=None):
        data = read_csv(infile, header=header, sep=sep)
        data = data.dropna()
        self.all_mut = [set() for i in range(nb_var)]
        self.mut_to_index = [None for i in range(nb_var)]
        for i in range(nb_var):
            self.all_mut[i] = list(set(data.iloc[:, i]))
            self.all_mut[i].sort()
            self.mut_to_index[i] = {k: j for j, k in enumerate(self.all_mut[i])}

        # number of mutations per variables
        self.nb_val = [len(el) for el in self.mut_to_index]
        for i in range(nb_var):
            data.iloc[:, i] = data.iloc[:, i].map(self.mut_to_index[i])

        self.data = tensor(data.to_numpy(dtype=float))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


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


def train(model, data, lr=1e-3, nb_epoch=100, bsize=32, wei_dec=1e-4, pen_emb=0):
    """
    Train a given model on the specified dataset.

    Args:
        model (nn.Module): The neural network model to train.
        data (Dataset): The dataset to train the model on.
        const (dict, optional): Constraints for the model. Defaults to None.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        nb_epoch (int, optional): Number of training epochs. Defaults to 100.
        bsize (int, optional): Batch size for training. Defaults to 32.
        wei_dec (float, optional): Weight decay for the optimizer. Defaults to 1e-3.
        val_data (Dataset, optional): Validation dataset. Defaults to None.

    Returns:
        list: A list of tuples containing mean batch loss and validation loss.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wei_dec)
    loss_f = GaussianNLLLoss()
    losses = []
    for _ in range(nb_epoch):
        loss_b, loss_l = [], []
        loader = DataLoader(data, batch_size=bsize, shuffle=True)
        for bi, batch in enumerate(loader):
            optimizer.zero_grad()
            # use the simple loss
            pfit, var, lat = model(batch[:, :-1].long())
            loss_mse = loss_f(pfit, batch[:, [-1]], var)
            if pen_emb > 0:
                loss_dist = (sum(torch.norm(el, p=2) for el in model.genes)/len(model.genes))**2
                loss = loss_mse + pen_emb * loss_dist
            else:
                loss = loss_mse + pen_emb
            loss.backward()
            optimizer.step()
            loss_b += [loss_mse.item()]
        losses += [mean(loss_b)]
    return losses


def train_reg(model, data, lr=1e-3, nb_epoch=100, bsize=32, wei_dec=1e-3, val_data=None):
    """
    Train a regression model on the specified dataset.

    Args:
        model (nn.Module): The regression model to train.
        data (Dataset): The dataset to train the model on.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        nb_epoch (int, optional): Number of training epochs. Defaults to 100.
        bsize (int, optional): Batch size for training. Defaults to 32.
        wei_dec (float, optional): Weight decay for the optimizer. Defaults to 1e-3.
        val_data (Dataset, optional): Validation dataset. Defaults to None.

    Returns:
        list: A list of tuples containing mean batch loss and validation loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wei_dec)
    losses = []
    for _ in range(nb_epoch):
        loss_d, loss_b = [], []
        loader = DataLoader(data, batch_size=bsize, shuffle=True)
        for bi, batch in enumerate(loader):
            optimizer.zero_grad()
            # use the simple loss
            pfit = model(batch[:, :-1].long())
            loss_mse = ((pfit - batch[:, [-1]])**2).mean()

            loss = loss_mse
            loss.backward()
            optimizer.step()
            loss_b += [loss_mse.item()]
            loss_d += [0]
        if val_data is not None:
            with no_grad():
                pfit = model(val_data[:, :-1].long())
                loss_mse_v = ((pfit - val_data[:, [-1]])**2).mean()
            losses += [(mean(loss_b), loss_mse_v)]
        else:
            losses += [(mean(loss_b), 0)]
    return losses


