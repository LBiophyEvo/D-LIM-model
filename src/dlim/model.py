from torch import nn, cat as tcat, tensor, save as tsave, load as tload, no_grad, zeros
from torch import normal, rand, exp, log, randn, arange, sin, cos, matmul, normal
import torch.nn.init as init
from numpy import sqrt, linspace, meshgrid, concatenate, newaxis, polyfit, polyval
from dlim.layers import Block 
from dlim.utils import spectral_init 
from dlim.dataset import Data_model

class DLIM(nn.Module):
    """
    Deep Latent Interaction Model (DLIM) for modeling interactions between genetic variables.

    Attributes:
        genes_emb (nn.ParameterList): Embeddings for each gene/variable.
        predictor (Block): Neural network block for prediction.
        conversion (list): Polynomial coefficients for phenotype conversion.
        spectral_init (SpectralInit): Utility for spectral initialization.

    Args:
        n_variables (list[int]): Number of states per variable.
        hid_dim (int, optional): Hidden layer size in predictor block. Defaults to 128.
        nb_layer (int, optional): Number of layers in predictor block. Defaults to 0.
        emb_init (list[torch.Tensor], optional): Initial embeddings for genes. Defaults to None.
        gap_thres (list[float], optional): Thresholds for spectral gap. Defaults to [0.01, 0.95].
        dropout_ratio (float, optional): Dropout ratio for predictor block. Defaults to 0.2.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
    """

    def __init__(self, n_variables, hid_dim=128, nb_layer=0, emb_init=None, gap_thres: list = [0.01, 0.95], dropout_ratio = 0.2, batch_norm = False):
        super(DLIM, self).__init__()

        self.gap_thres = gap_thres
        self.n_variables = n_variables

        # Initialize gene embeddings: use provided or random (Xavier/normal)
        if emb_init is not None:
            self.genes_emb = nn.ParameterList([nn.Parameter(el) for el in emb_init])
        else:
            self.genes_emb = nn.ParameterList([nn.Parameter(randn((nb, 1))) for nb in n_variables])
        if emb_init is None:
            for el in self.genes_emb:
                init.xavier_normal_(el)

        # Predictor block: processes concatenated embeddings
        self.predictor = Block(len(self.genes_emb), 2, hid_dim, nb_layer, dropout_ratio = dropout_ratio, batch_norm = batch_norm)
        self.conversion = [None for _ in self.genes_emb]
        self.spectral_init = spectral_init()

    def forward(self, gene, pre_lat=False):
        """
        Forward pass through DLIM model.

        Args:
            gene (Tensor): Input gene indices or latent representation.
            pre_lat (bool): If True, use gene as latent directly.

        Returns:
            mu (Tensor): Predicted mean.
            var (Tensor): Predicted variance (exponentiated).
            lat (Tensor): Latent representation.
        """
        if not pre_lat:
            # Gather embeddings for each gene and concatenate
            lat = tcat([self.genes_emb[i][gene[:, i]] for i in range(len(self.genes_emb))], dim=1)
        else:
            lat = gene
        fit = self.predictor(lat)
        mu, var = fit[:, [0]], fit[:, [1]]
        return mu, exp(var), lat

    def train_convert(self, genes, pheno, variable):
        """
        Fit a polynomial to map phenotype values to gene embeddings for a variable.

        Args:
            genes (array-like): Indices of genes.
            pheno (array-like): Phenotype values.
            variable (int): Variable index.
        """
        self.conversion[variable] = polyfit(pheno, self.genes_emb[variable][genes].detach(), 3)

    def spec_init_emb(self, data: Data_model, sim="pearson", temp=1., force=True):
        """
        Apply spectral initialization to gene embeddings.

        Args:
            data (Data_model): Data model instance.
            sim (str): Similarity measure for spectral initialization.
            temp (float): Temperature for similarity measure.
            force (bool): Force spectral initialization if gap is within threshold.
        """
        emb_init = []
        for c, nb in enumerate(self.n_variables):
            # Compute correlation matrix and Fiedler vector
            cov_mat = self.spectral_init.compute_cor_scores(data, col=c, sim_type=sim, temperatue=temp)
            fiedler_vec, eig_val = self.spectral_init.calculate_fiedler_vector(cov_mat, eig_val=True)
            # Compute the gap between the first and second eigen values
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
        """
        Update gene embeddings for a variable using polynomial conversion.

        Args:
            genes (array-like): Indices of genes to update.
            pheno (array-like): Phenotype values.
            variable (int): Variable index.
        """
        self.genes_emb[variable].data[genes] = tensor(polyval(self.conversion[variable], pheno),
                                                  dtype=self.genes_emb[variable].dtype).reshape(-1, 1)