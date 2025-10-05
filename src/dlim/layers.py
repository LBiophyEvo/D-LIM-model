from torch import nn, cat as tcat, tensor, save as tsave, load as tload, no_grad, zeros
import torch.nn.init as init

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

    def __init__(self, in_d, out_d, hid_d, nb_layer=0, dropout_ratio = 0.2, batch_norm = False):
        super(Block, self).__init__()
        self.dropout = nn.Dropout(dropout_ratio)
        if batch_norm:
            self.pred = nn.ModuleList([nn.Linear(in_d, hid_d), nn.ReLU(), self.dropout])
        else:
            self.pred = nn.ModuleList([nn.Linear(in_d, hid_d), nn.BatchNorm1d(hid_d), nn.ReLU(), self.dropout])


        for _ in range(nb_layer):
            if batch_norm:
                self.pred += [nn.Linear(hid_d, hid_d), nn.BatchNorm1d(hid_d), nn.ReLU(), self.dropout]
            else:
                self.pred += [nn.Linear(hid_d, hid_d), nn.ReLU(), self.dropout]

        self.pred += [nn.Linear(hid_d, out_d)]

        for el in self.pred:
            if isinstance(el, nn.Linear):
                init.xavier_normal_(el.weight)

    def forward(self, x):
        for el in self.pred:
            x = el(x)
        return x