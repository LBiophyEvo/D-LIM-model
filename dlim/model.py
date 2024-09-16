from torch import nn, cat as tcat, tensor, save as tsave, load as tload, no_grad, zeros
from torch import normal, rand, exp, log, randn, arange, sin, cos, matmul, normal
import torch.nn.init as init
from numpy import sqrt, linspace, meshgrid, concatenate, newaxis, polyfit, polyval
from dlim.layers import Block 
from dlim.utils import spectral_init 
from dlim.dataset import Data_model


class DLIM(nn.Module):
    """
    Deep Latent Interaction Model (DLIM) for handling interactions between different variables.

    Attributes:
        genes_emb (nn.ParameterList): A list of parameters representing genes embeddings.
        predictor (Block): A neural network block for processing.
        conversion (list): List of polynomial coefficients for conversion of genes.
        spectral_init (SpectralInit): An instance for spectral initialization.

    Args:
        n_variables (list[int]): The number of states per variable.
        hid_dim (int, optional): The size of the hidden layer in the `predictor` block. Defaults to 128.
        nb_layer (int, optional): The number of layers in the `predictor` block. Defaults to 0.
        emb_init (list[torch.Tensor], optional): Initial embeddings for the genes. Defaults to None.
        gap_thres (list[float], optional): Thresholds for determing if we use spectral initialization. Defaults to [0.01, 0.95].
    """

    def __init__(self, n_variables, hid_dim=128, nb_layer=0, emb_init=None, gap_thres: list = [0.01, 0.95]):
        super(DLIM, self).__init__()

        self.gap_thres = gap_thres
        self.n_variables = n_variables
        if emb_init is not None:
            self.genes_emb = nn.ParameterList([nn.Parameter(el) for el in emb_init])
        else:
            self.genes_emb = nn.ParameterList([nn.Parameter(randn((nb, 1))) for nb in n_variables])
        if emb_init is None:
            for el in self.genes_emb:
                init.xavier_normal_(el)
        self.predictor = Block(len(self.genes_emb), 2, hid_dim, nb_layer)
        self.conversion = [None for _ in self.genes_emb]
        self.spectral_init = spectral_init()

    def forward(self, gene, pre_lat=False):
        if not pre_lat:
            lat = tcat([self.genes_emb[i][gene[:, i]] for i in range(len(self.genes_emb))], dim=1)
        else:
            lat = gene
        fit = self.predictor(lat)
        mu, var = fit[:, [0]], fit[:, [1]]
    
        return mu, exp(var), lat


    def train_convert(self, genes, pheno, variable):
        "gene = id; pheno = float; variable = variable id"
        self.conversion[variable] = polyfit(pheno, self.genes_emb[variable][genes].detach(), 3)

    def spec_init_emb(self, data: Data_model, sim="pearson", temp=1., force=True):
        """
        Apply spectral initialization to embeddings.

        Args:
            data (Data_model): DataModel instance.
            sim (str): Similarity measure for spectral initialization.
            temp (float): Temperature for the similarity measure.
            force (bool): Whether to force spectral initialization.
        """
        emb_init = []
        for c, nb in enumerate(self.n_variables):
            cov_mat = self.spectral_init.compute_cor_scores(data, col=c, sim_type=sim, temperatue=temp)
            fiedler_vec, eig_val = self.spectral_init.calculate_fiedler_vector(cov_mat, eig_val=True)
            spec_gap = (eig_val[1]-eig_val[0])
            if force and (spec_gap < self.gap_thres[1] and spec_gap > self.gap_thres[0]):
                print(f"spectral gap = {spec_gap}")
                emb_init += [nn.Parameter(fiedler_vec.reshape(-1, 1))]
            else:
                print(f"spectral gap = {spec_gap}, so we initialize phenotypes randomly")
                emb_init += [nn.Parameter(randn((nb, 1)))]
                init.xavier_normal_(emb_init[-1])
        self.genes_emb = nn.ParameterList(emb_init)

    def update_emb(self, genes, pheno, variable):
        self.genes_emb[variable].data[genes] = tensor(polyval(self.conversion[variable], pheno),
                                                  dtype=self.genes_emb[variable].dtype).reshape(-1, 1)



