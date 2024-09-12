
import torch 
import numpy as np 
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine, euclidean
from scipy.linalg import eigh
from dlim.dataset import Data_model


class spectral_init():
    def __init__(self):
        pass

    def calculate_fiedler_vector(self, A, eig_val=False):
        """Compute the spectral initialization.
        - A is the adjacency matrix
        """

        if not isinstance(A, torch.Tensor):
            A = torch.Tensor(A)
        if A.min() <= 0.:
            A = A - A.min()
            A += 1e-10              # numerical stability
        D_v = A.sum(axis=1)
        D = torch.diag(D_v)
        L = D - A
        D_is = torch.diag(1.0 / (D_v**0.5))
        L = D_is @ L @ D_is

        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        fiedler_vector = eigenvectors[:, 1]
        fiedler_vector = (fiedler_vector - fiedler_vector.mean())/fiedler_vector.std()
        if eig_val:
            return fiedler_vector, eigenvalues
        else:
            return fiedler_vector


    def compute_cor_scores(self, data: Data_model, col: int , sim_type: str = 'pearson', temperatue: float = 1.0):
        
        all_var = data.substitutions_tokens[col]
        nb_var = len(data.substitutions_tokens[col])
        train_data = data.data
        cov_mat = np.zeros((nb_var, nb_var))
        col_d = [i for i in range(train_data.shape[1]-1) if i!=col]

        for _, i in all_var.items():
            for _, j in all_var.items():

                if i <= j:
                    di = train_data[train_data[:, col] == float(i)]
                    dj = train_data[train_data[:, col] == float(j)]
                    # Convert rows into a structured array format for element-wise comparison
                    di_sub = di[:, col_d]
                    dj_sub = dj[:, col_d]
                    # Find matching rows by broadcasting and comparing all rows
                    matches = (di_sub[:, None] == dj_sub).all(axis=2)
                    # Get the indices of matches
                    a1_idx, a2_idx = np.where(matches)

                    if a1_idx.shape[0] > 2:
                        if sim_type == "pearson":
                            cov_mat[i, j] = pearsonr(di[a1_idx, -1], dj[a2_idx, -1])[0]
                        elif sim_type == "cosine":
                            cov_mat[i, j] = 1 - cosine(di[a1_idx, -1], dj[a2_idx, -1])
                        elif sim_type == "euclidean":
                            dist = ((di[a1_idx, -1]-dj[a2_idx, -1])**2).mean()
                            cov_mat[i, j] = np.exp(- dist/temperatue)
                        else:
                            raise ValueError("Incorrect similarity type proposed")
                    else:
                        cov_mat[i, j] = 0.
                    cov_mat[j, i] = cov_mat[i, j]
        return  cov_mat
