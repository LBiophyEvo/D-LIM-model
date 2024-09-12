from torch.utils.data import Dataset, DataLoader
from pandas import DataFrame
from torch import tensor, Tensor

class Data_model(Dataset):
    """
    A custom dataset class for handling data in machine learning models.

    Attributes:
        substitutions (List[List[str]]): List of unique mutations for each variable.
        substitutions_tokens (List[Dict[str, int]]): List of dictionaries mapping mutations to indices for each variable.
        data (Tensor): Tensor representation of the encoded data.
        nb_val (List[int]): Number of unique substitutions (mutations) per variable.

    Args:
        data (pd.DataFrame): Input data as a pandas DataFrame.
        n_variables (int): Number of variables (columns) in the dataset.
    """
    def __init__(self, data: DataFrame, n_variables: int):
        data = data.dropna()
        self.substitutions = [set() for i in range(n_variables)]
        self.substitutions_tokens = [None for i in range(n_variables)]
        for i in range(n_variables):
            self.substitutions[i] = list(set(data.iloc[:, i]))
            self.substitutions[i].sort()
            self.substitutions_tokens[i] = {k: j for j, k in enumerate(self.substitutions[i])}

        # number of substitions per variables
        self.nb_val = [len(el) for el in self.substitutions_tokens]
        for i in range(n_variables):
            data.iloc[:, i] = data.iloc[:, i].map(self.substitutions_tokens[i])

        self.data = tensor(data.to_numpy(dtype=float))


    def subset(self, IDX):
        return SubDataset(self, IDX)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    



class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset: Data_model, indices):
        self.data = dataset.data[indices]
        self.substitutions = dataset.substitutions
        self.substitutions_tokens = dataset.substitutions_tokens
        self.nb_val = dataset.nb_val


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)