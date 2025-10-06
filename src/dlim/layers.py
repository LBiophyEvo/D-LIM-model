from torch import nn, relu, dropout, cat as tcat
import torch.nn.init as init

class Block(nn.Module):
    """
    A neural network block used in DLIM for phenotype prediction.

    This block consists of a configurable number of hidden layers, each followed by
    an activation function and optional dropout and batch normalization.

    Attributes:
        layers (nn.ModuleList): List of hidden layers.
        out_layer (nn.Linear): Output layer mapping to final prediction.
        dropout_ratio (float): Dropout probability.
        batch_norm (bool): Whether to use batch normalization.

    Args:
        in_dim (int): Input dimension (number of features).
        out_dim (int): Output dimension (number of targets).
        hid_dim (int): Hidden layer size.
        nb_layer (int): Number of hidden layers.
        dropout_ratio (float, optional): Dropout probability. Defaults to 0.2.
        batch_norm (bool, optional): Use batch normalization. Defaults to False.
    """
    def __init__(self, in_dim, out_dim, hid_dim, nb_layer, dropout_ratio=0.2, batch_norm=False):
        super(Block, self).__init__()
        self.dropout = nn.Dropout(dropout_ratio)
        # Add input layer & batch normalization 
        if batch_norm:
            self.pred = nn.ModuleList([nn.Linear(in_dim, hid_dim), nn.ReLU(), self.dropout])
        else:
            self.pred = nn.ModuleList([nn.Linear(in_dim, hid_dim), nn.BatchNorm1d(hid_dim), nn.ReLU(), self.dropout])

        # Add hidden layer 
        for _ in range(nb_layer):
            if batch_norm:
                self.pred += [nn.Linear(hid_dim, hid_dim), nn.BatchNorm1d(hid_dim), nn.ReLU(), self.dropout]
            else:
                self.pred += [nn.Linear(hid_dim, hid_dim), nn.ReLU(), self.dropout]
       
        # Add output layer

        self.pred += [nn.Linear(hid_dim, out_dim)]

        # Normalize each layer 
        for el in self.pred:
            if isinstance(el, nn.Linear):
                init.xavier_normal_(el.weight)

    def forward(self, x):
        """
        Forward pass through the block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output predictions.
        """
        for layer in self.pred:
            x = layer(x)
        return x