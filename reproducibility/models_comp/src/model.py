from torch import nn, cat as tcat, tensor, save as tsave, load as tload, no_grad, zeros
from torch import normal, rand, exp, log, randn, arange
from torch import float32 as tfloat, ones_like
from numpy import sqrt, linspace, meshgrid, concatenate, newaxis, polyfit, polyval
from matplotlib.colors import TwoSlopeNorm
import torch.nn.init as init
import torch
import warnings


class Block(nn.Module):
    """
    Neural network block with a configurable number of hidden layers.
    Each hidden layer uses ReLU activation. The block maps from input
    dimension to output dimension via hidden layers.

    Attributes:
        pred (nn.ModuleList): List of layers in the block.

    Args:
        in_d (int): Input dimension
        out_d (int): Output dimension
        hid_d (int): Hidden dimension
        nb_layer (int): Number of hidden layers
    """

    def __init__(self, in_d, out_d, hid_d, nb_layer=0):
        super(Block, self).__init__()

        # Input layer + activation
        self.pred = nn.ModuleList([nn.Linear(in_d, hid_d), nn.ReLU()])
        # Hidden layers with ReLU
        for _ in range(nb_layer):
            self.pred += [nn.Linear(hid_d, hid_d), nn.ReLU()]
        # Output layer
        self.pred += [nn.Linear(hid_d, out_d)]

        # Xavier initialization for linear layers
        for el in self.pred:
            if isinstance(el, nn.Linear):
                init.xavier_normal_(el.weight)

    def forward(self, x):
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        for el in self.pred:
            x = el(x)
        return x


class Add_Latent(nn.Module):
    """
    Additive latent model.

    Each input variable has a learnable latent embedding.
    Latent embeddings are summed and passed through an MLP to predict
    mean and variance of the output.

    Args:
        nb_var (int or list): Number of variables or list of variable sizes
        nb_state (int): Number of latent states per variable
        emb (int): Latent embedding dimension
        hid (int): Hidden dimension of MLP
        nb_layer (int): Number of hidden layers in MLP
    """

    def __init__(self, nb_var, nb_state=5, emb=1, hid=128, nb_layer=0):
        super(Add_Latent, self).__init__()

        self.nb_var = nb_var
        # Initialize latent embeddings for each variable
        if isinstance(nb_var, int):
            self.genes = nn.ParameterList([nn.Parameter(randn((nb_state, emb))) for _ in range(nb_var)])
        else:
            self.genes = nn.ParameterList([nn.Parameter(randn((nb, emb))) for nb in nb_var])

        # Xavier initialization for latent embeddings
        for el in self.genes:
            init.xavier_normal_(el)

        # MLP to predict interactions
        self.epi = Block(emb, 2, hid, nb_layer)

    def forward(self, gene, pre_lat=False, detach=False):
        """
        Forward pass through the additive latent model.

        Args:
            gene (torch.Tensor): Input tensor of gene indices or latent embeddings
            pre_lat (bool): If True, `gene` is treated as latent embeddings
            detach (bool): If True, outputs are detached from computation graph

        Returns:
            mu (torch.Tensor): Predicted mean
            var (torch.Tensor): Predicted variance (exp of MLP output)
            lat (torch.Tensor): Latent representation
        """
        # Get latent embeddings for each gene
        if not pre_lat:
            lat = tcat([self.genes[i][gene[:, [i]]] for i in range(len(self.genes))], dim=1)
        else:
            lat = gene

        # Sum latent embeddings across variables and pass through MLP
        fit = self.epi(lat.sum(dim=1))
        mu, var = fit[:, [0]], fit[:, [1]]

        if detach:
            return mu.detach(), exp(var).detach(), lat.detach()
        else:
            return mu, exp(var), lat


class Regression(nn.Module):
    """
    Simple linear additive regression model.

    Each variable has a learnable latent vector. Predicted output
    is the sum of latent contributions from each variable.

    Args:
        nb_var (int): Number of variables
        nb_state (int): Number of latent states per variable
    """

    def __init__(self, nb_var, nb_state=5):
        super(Regression, self).__init__()

        self.nb_var = nb_var
        # Latent parameters for each variable
        self.genes = nn.Parameter(rand((nb_var, nb_state, 1)))

    def forward(self, gene):
        """
        Forward pass for linear regression.

        Args:
            gene (torch.Tensor): Input tensor of gene indices

        Returns:
            fit (torch.Tensor): Predicted fitness
        """
        # Lookup latent vectors
        lat = self.genes[arange(0, gene.shape[1]), gene]
        # Sum across variables
        fit = lat.view(gene.shape[0], -1).sum(dim=-1).view(-1, 1)
        return fit
