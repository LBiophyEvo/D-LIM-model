from torch import nn, cat as tcat, tensor, save as tsave, load as tload, no_grad, zeros
from torch import normal, rand, exp, log, randn, arange, sin, cos, matmul, normal
from torch import float32 as tfloat, cat, ones_like
from numpy import sqrt, linspace, meshgrid, concatenate, newaxis, polyfit, polyval
from matplotlib.colors import TwoSlopeNorm
import torch.nn.init as init
import torch
import warnings


class Block(nn.Module):
    """
    Represents a neural network block consisting of a sequence of linear layers
    and ReLU activations. The number of layers is configurable.

    Attributes:
        pred (nn.ModuleList): A list of layers in the block.

    Args:
        in_d (int): The input dimension.
        out_d (int): The output dimension.
        hid_d (int): The hidden dimension.
        nb_layer (int, optional): The number of layers in the block. Defaults to 0.

    Methods:
        forward(x): Defines the forward pass of the block.
    """

    def __init__(self, in_d, out_d, hid_d, nb_layer=0):
        super(Block, self).__init__()

        self.pred = nn.ModuleList([nn.Linear(in_d, hid_d), nn.ReLU()])
        for _ in range(nb_layer):
            self.pred += [nn.Linear(hid_d, hid_d), nn.ReLU()]
        self.pred += [nn.Linear(hid_d, out_d)]

        for el in self.pred:
            if isinstance(el, nn.Linear):
                init.xavier_normal_(el.weight)

    def forward(self, x):
        for el in self.pred:
            x = el(x)
        return x


class Add_Latent(nn.Module):
    """Simple additive latent model
    """

    def __init__(self, nb_var, nb_state=5, emb=1, hid=128, nb_layer=0):
        super(Add_Latent, self).__init__()

        self.nb_var = nb_var
        if type(nb_var) is int:
            self.genes = nn.ParameterList([nn.Parameter(randn((nb_state, emb))) for nb in range(nb_var)])
        else:
            self.genes = nn.ParameterList([nn.Parameter(randn((nb, emb))) for nb in nb_var])

        for el in self.genes:
            init.xavier_normal_(el)
        self.epi = Block(emb, 2, hid, nb_layer)


    def forward(self, gene, pre_lat=False, detach=False):
        if not pre_lat:
            lat = tcat([self.genes[i][gene[:, [i]]] for i in range(len(self.genes))], dim=1)
        else:
            lat = gene
        fit = self.epi(lat.sum(dim=1))
        mu, var = fit[:, [0]], fit[:, [1]]
        if detach:
            return mu.detach(), exp(var).detach(), lat.detach()
        else:
            return mu, exp(var), lat

class Regression(nn.Module):

    def __init__(self, nb_var, nb_state=5):
        super(Regression, self).__init__()

        self.nb_var = nb_var
        self.genes = nn.Parameter(rand((nb_var, nb_state, 1)))

    def forward(self, gene):
        lat = self.genes[arange(0, gene.shape[1]), gene]
        fit = lat.view(gene.shape[0], -1).sum(dim=-1).view(-1, 1)
        return fit
