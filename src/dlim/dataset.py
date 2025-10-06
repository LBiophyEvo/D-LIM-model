from torch.utils.data import Dataset, DataLoader
from pandas import DataFrame
from torch import tensor, Tensor

class Data_model(Dataset):
    """
    A custom dataset class for handling genotype-phenotype data in machine learning models.

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
        # Remove any rows with missing values
        data = data.dropna()
        self.substitutions = [set() for i in range(n_variables)]
        self.substitutions_tokens = [None for i in range(n_variables)]
        # Build substitution lists and token mappings for each variable
        for i in range(n_variables):
            self.substitutions[i] = list(set(data.iloc[:, i]))
            self.substitutions[i].sort()
            self.substitutions_tokens[i] = {k: j for j, k in enumerate(self.substitutions[i])}

        # Store the number of unique substitutions per variable
        self.nb_val = [len(el) for el in self.substitutions_tokens]
        # Map each substitution to its token index
        for i in range(n_variables):
            data.iloc[:, i] = data.iloc[:, i].map(self.substitutions_tokens[i])

        # Store the encoded data as a tensor
        self.data = tensor(data.to_numpy(dtype=float))

    def subset(self, IDX):
        """
        Create a subset of the dataset using the provided indices.

        Args:
            IDX (list or array): Indices to select.

        Returns:
            SubDataset: A subset dataset object.
        """
        return SubDataset(self, IDX)

    def __getitem__(self, index):
        """
        Retrieve a single data point.

        Args:
            index (int): Index of the data point.

        Returns:
            Tensor: The encoded data for the given index.
        """
        return self.data[index]

    def __len__(self):
        """
        Get the total number of data points.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)
    

class SubDataset:
    """
    Subsets a `Data_model` dataset by selecting the examples given by `indices`.

    Attributes:
        data (Tensor): Subset of the original data.
        substitutions (List[List[str]]): Unique mutations for each variable.
        substitutions_tokens (List[Dict[str, int]]): Mapping mutations to indices.
        nb_val (List[int]): Number of unique substitutions per variable.

    Args:
        dataset (Data_model): The original dataset to subset.
        indices (list or array): Indices to select for the subset.
    """

    def __init__(self, dataset: Data_model, indices):
        # Select the subset of data using the provided indices
        self.data = dataset.data[indices]
        self.substitutions = dataset.substitutions
        self.substitutions_tokens = dataset.substitutions_tokens
        self.nb_val = dataset.nb_val

    def __getitem__(self, index):
        """
        Retrieve a single data point from the subset.

        Args:
            index (int): Index of the data point.

        Returns:
            Tensor: The encoded data for the given index.
        """
        return self.data[index]

    def __len__(self):
        """
        Get the total number of data points in the subset.

        Returns:
            int: Number of samples in the subset.
        """
        return len(self.data)