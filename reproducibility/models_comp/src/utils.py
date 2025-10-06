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
    Custom PyTorch Dataset for handling tabular mutation/fitness data.

    Each variable/column in the data is mapped to integer indices corresponding 
    to unique states. Supports use with models requiring PyTorch tensors.

    Attributes:
        all_mut (list of lists): Unique values for each variable.
        mut_to_index (list of dicts): Maps each value of a variable to its integer index.
        data (torch.Tensor): Tensor representation of the dataset.
        nb_val (list): Number of unique values per variable.

    Args:
        infile (str): Path to the CSV file containing the data.
        nb_var (int): Number of variables in the dataset.
        sep (str, optional): Column separator in the CSV file. Defaults to ",".
        header (int or None, optional): Row to use as column names. Defaults to None.
    """

    def __init__(self, infile, nb_var, sep=",", header=None):
        # Read CSV file and drop missing values
        data = read_csv(infile, header=header, sep=sep)
        data = data.dropna()

        # Initialize lists to store unique mutations and mapping to indices
        self.all_mut = [set() for _ in range(nb_var)]
        self.mut_to_index = [None for _ in range(nb_var)]

        # Identify unique states for each variable and create mappings
        for i in range(nb_var):
            self.all_mut[i] = list(set(data.iloc[:, i]))
            self.all_mut[i].sort()  # Ensure consistent ordering
            self.mut_to_index[i] = {k: j for j, k in enumerate(self.all_mut[i])}

        # Number of unique states per variable
        self.nb_val = [len(el) for el in self.mut_to_index]

        # Convert data to integer indices
        for i in range(nb_var):
            data.iloc[:, i] = data.iloc[:, i].map(self.mut_to_index[i])

        # Convert to torch tensor for use in models
        self.data = tensor(data.to_numpy(dtype=float))

    def __getitem__(self, index):
        """
        Retrieve a single data point.

        Args:
            index (int): Index of the data point.

        Returns:
            torch.Tensor: Data at the given index.
        """
        return self.data[index]

    def __len__(self):
        """
        Returns:
            int: Total number of data points.
        """
        return len(self.data)


def dist_loss(lat, const, wei=1):
    """
    Compute a distance-based loss for latent variables with optional constraints.

    Args:
        lat (torch.Tensor): Latent representations of data (N x D).
        const (dict): Dictionary of constraint weights for specific variable groups.
        wei (float, optional): Scaling factor for the total constraint loss. Defaults to 1.

    Returns:
        torch.Tensor: Weighted distance loss.
    """
    losses = []
    # Iterate over variable groups in constraints
    for pi in const:
        mat = lat[pi]  # Select latent embeddings for group
        # Compute pairwise squared distances
        dist = ((mat[:, newaxis, :] - mat[newaxis, :, :])**2).mean(dim=-1)
        # Weight distances by constraints and average
        losses += [(dist * const[pi]).mean()]
    return wei * sum(losses) / len(losses)


def train(model, data, lr=1e-3, nb_epoch=100, bsize=32, wei_dec=1e-4, pen_emb=0):
    """
    Train a neural network model with Gaussian Negative Log-Likelihood loss.

    Args:
        model (nn.Module): PyTorch model to train.
        data (Dataset): Training dataset.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        nb_epoch (int, optional): Number of epochs. Defaults to 100.
        bsize (int, optional): Batch size. Defaults to 32.
        wei_dec (float, optional): Weight decay for optimizer. Defaults to 1e-4.
        pen_emb (float, optional): L2 penalty on embeddings. Defaults to 0.

    Returns:
        list: Mean batch loss for each epoch.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wei_dec)
    loss_f = GaussianNLLLoss()
    losses = []

    for _ in range(nb_epoch):
        loss_b = []
        loader = DataLoader(data, batch_size=bsize, shuffle=True)

        for batch in loader:
            optimizer.zero_grad()

            # Forward pass: returns predicted mean, variance, and latent variables
            pfit, var, lat = model(batch[:, :-1].long())
            loss_mse = loss_f(pfit, batch[:, [-1]], var)

            # Optional L2 penalty on embeddings
            if pen_emb > 0:
                loss_dist = (sum(torch.norm(el, p=2) for el in model.genes) / len(model.genes))**2
                loss_total = loss_mse + pen_emb * loss_dist
            else:
                loss_total = loss_mse

            loss_total.backward()
            optimizer.step()
            loss_b.append(loss_mse.item())

        # Average batch loss for this epoch
        losses.append(mean(loss_b))
    return losses


def train_reg(model, data, lr=1e-3, nb_epoch=100, bsize=32, wei_dec=1e-3, val_data=None):
    """
    Train a simple regression model using MSE loss.

    Args:
        model (nn.Module): Regression model to train.
        data (Dataset): Training dataset.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        nb_epoch (int, optional): Number of epochs. Defaults to 100.
        bsize (int, optional): Batch size. Defaults to 32.
        wei_dec (float, optional): Weight decay for optimizer. Defaults to 1e-3.
        val_data (torch.Tensor, optional): Validation data for monitoring. Defaults to None.

    Returns:
        list: List of tuples (train_loss, val_loss) per epoch.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wei_dec)
    losses = []

    for _ in range(nb_epoch):
        loss_b, loss_d = [], []
        loader = DataLoader(data, batch_size=bsize, shuffle=True)

        for batch in loader:
            optimizer.zero_grad()

            # Forward pass
            pfit = model(batch[:, :-1].long())
            loss_mse = ((pfit - batch[:, [-1]])**2).mean()

            loss_mse.backward()
            optimizer.step()

            loss_b.append(loss_mse.item())
            loss_d.append(0)

        # Compute validation loss if validation data is provided
        if val_data is not None:
            with no_grad():
                pfit = model(val_data[:, :-1].long())
                val_loss = ((pfit - val_data[:, [-1]])**2).mean()
            losses.append((mean(loss_b), val_loss))
        else:
            losses.append((mean(loss_b), 0))

    return losses
