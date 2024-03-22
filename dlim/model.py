from torch import nn, cat as tcat, tensor, save as tsave, load as tload, no_grad, zeros
from torch import normal, rand, exp, log, randn, arange, sin, cos, matmul, normal
from torch import float32 as tfloat, cat, ones_like
from numpy import sqrt, linspace, meshgrid, concatenate, newaxis, polyfit, polyval
import torch.nn.init as init
import torch


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


class DLIM(nn.Module):
    """
    Deep Latent Interaction Model (DLIM) for handling interactions between different variables.

    Attributes:
        nb_var (int): The number of variables.
        genes (nn.ParameterList): A list of parameters representing genes.
        epi (Block): A neural network block for processing.
        conversion (list): List of conversion factors for genes.

    Args:
        nb_var (int): The number of variables.
        nb_state (int, optional): The number of states for each variable. Defaults to 5.
        emb (int, optional): The size of the embedding. Defaults to 1.
        hid (int, optional): The size of the hidden layer in the `epi` block. Defaults to 128.
        nb_layer (int, optional): The number of layers in the `epi` block. Defaults to 0.

    Methods:
        forward(gene, pre_lat=False, detach=False): Defines the forward pass of DLIM.
        train_convert(genes, pheno, variable): Trains the conversion for a given variable.
        update_emb(genes, pheno, variable): Updates the embedding for a given variable.
        plot(ax, data=None): Plots the model predictions.
    """

    def __init__(self, nb_var, nb_state=5, emb=1, hid=128, nb_layer=0):
        super(DLIM, self).__init__()

        self.nb_var = nb_var
        if type(nb_var) is int:
            self.genes = nn.ParameterList([nn.Parameter(randn((nb_state, emb))) for nb in range(nb_var)])
        else:
            self.genes = nn.ParameterList([nn.Parameter(randn((nb, emb))) for nb in nb_var])
        for el in self.genes:
            init.xavier_normal_(el)
        self.epi = Block(len(self.genes)*emb, 2, hid, nb_layer)
        self.conversion = [None for _ in self.genes]

    def forward(self, gene, pre_lat=False, detach=False):
        if not pre_lat:
            lat = tcat([self.genes[i][gene[:, i]] for i in range(len(self.genes))], dim=1)
        else:
            lat = gene
        fit = self.epi(lat)
        mu, var = fit[:, [0]], fit[:, [1]]
        if detach:
            return mu.detach(), exp(var).detach(), lat.detach()
        else:
            return mu, exp(var), lat

    def train_convert(self, genes, pheno, variable):
        "gene = id; pheno = float; variable = variable id"
        self.conversion[variable] = polyfit(pheno, self.genes[variable][genes].detach(), 3)

    def update_emb(self, genes, pheno, variable):
        self.genes[variable].data[genes] = tensor(polyval(self.conversion[variable], pheno),
                                                  dtype=self.genes[variable].dtype).reshape(-1, 1)

    def plot(self, ax, data=None):
        "only for pairs"
        min_x, max_x = self.genes[0].min().item(), self.genes[0].max().item()
        delta_x = 0.1*(max_x - min_x)
        min_y, max_y = self.genes[1].min().item(), self.genes[1].max().item()
        delta_y = 0.1*(max_y - min_y)
        x_v = linspace(min_x - delta_x, max_x + delta_x, 200)
        y_v = linspace(min_y - delta_y, max_y + delta_y, 200)
        x_m, y_m = meshgrid(x_v, y_v)
        data_np = concatenate((x_m[newaxis, :, :], y_m[newaxis, :, :]), axis=0)
        data_m = tensor(data_np).transpose(0, 2).reshape(-1, 2).to(tfloat)
        pred_l = self.epi(data_m)[:, [0]].detach().numpy().reshape(200, 200).T

        ax.contourf(x_m, y_m, pred_l, cmap="bwr", alpha=0.4)
        ax.set_xlabel("$Z^1$")
        ax.set_ylabel("$Z^2$")

        if data is not None:
            fit, var, lat = self.forward(data[:, :-1].int(), detach=True)
            ax.scatter(lat[:, 0], lat[:, 1], c=data[:, -1], s=2, cmap="bwr", marker="x")


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
