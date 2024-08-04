from typing import Union, Optional
import warnings
import torch

from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data.data import Data

from transform import standard_normal

#########################################################################################################################
class SpectralConcatenation(Module):

    # performs the concatenation
    def __init__(self, spec_in, spec_out, num_linear=2, add_relu=True):

        # spec_in: input dimension for each linear layer, and also the number of dimension of egenvectors
        # spec_out: output dimension for each liner layer
        # num_linear: number of linear layer, default=2
        # add_relu: if True, will add relu after normalization

        super().__init__()

        if num_linear == 1:
            warnings.warn('number of linear layers for the spectral concatenation is 1, are you sure!...')
        if num_linear == 0:
            raise TypeError("number of linear layers for spectral concatenation is 0 \n it must be greater than 0")

        self.spec_in = spec_in
        self.spec_out = spec_out
        self.num_linear = num_linear
        self.add_relu = add_relu
        self.reduce = torch.nn.Parameter(torch.FloatTensor(num_linear, spec_in, spec_out))

        self.reset_parameters()

    ############################
    def reset_parameters(self):
        # performs xavier weight initialization

        # torch.nn.init.xavier_uniform_(self.reduce)
        for nl in range(self.num_linear):
            torch.nn.init.orthogonal_(self.reduce[nl])

    ############################
    @staticmethod
    def row_normalize(X: torch.tensor):
        # normalize row of X to 1

        return F.normalize(input=X, p=2.0, dim=2, eps=1e-12, out=None)

    ############################ forward
    def forward(self, X):

        # ModuleList can act as an iterable, or be indexed using ints
        # X: a matric spec_in time spec_out

        y = X @ self.reduce  # X: eigenvectors, has spec_in=2k dimension, self.reduce [num_linear, spec_in=2k, specout=k]
        y = self.row_normalize(y)
        y = torch.transpose(y, 0, 1)
        y = torch.flatten(y, start_dim=1)
        if self.add_relu:
            y1 = F.relu(y)
            y2 = F.relu(-y)
            return torch.hstack((y1, y2))
        return y

    ############################ __repr__
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.spec_in) + ' -> ' \
               + str(self.spec_out) + ', ' + \
               'num_linear=' + str(self.numlinear) + ', ' + \
               'add_relu=' + str(self.add_relu) + ')'


######################################################################################################################### convGCN
class DynamicGCNconv(Module):

    def __init__(
            self, 
            in_dim: int,
            hidden_dims: Union[tuple[int, int], list[int]],
            out_dim: int,
            transform_mode: str,
            mutate_x_epoch: bool = False,
            add_relu: bool = True,
            bias: bool = True,
            init: str = "xavier",
            normalize: bool = True,
            add_self_loops: Optional[bool] = None,
            **kwargs,
    ):

        super().__init__()
        
        if isinstance(hidden_dims, tuple):
            hidden_dims = [hidden_dims[1]] * hidden_dims[0]

        dims = [in_dim] + hidden_dims + [out_dim]

        self.in_dim: int = in_dim
        self.hidden_dims: list[int] = hidden_dims
        self.out_dim: int = out_dim
        self.transform_mode = transform_mode
        self.mutate_x_epoch = mutate_x_epoch
        self.add_relu: bool = add_relu
        self.bias: bool = bias
        self.init: str = init
        self.dims: list[int] = dims
        self.dropout: Optional[float] = kwargs['dropout'] if 'dropout' in kwargs else None
        self.normalize: bool = normalize
        self.add_self_loops: bool = add_self_loops

        ############################
        print(f"dims are {dims}")

        self.gcn_conv = ModuleList()
        for i in range(len(self.dims) - 1):
            self.gcn_conv.append(
                GCNConv(
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    add_self_loops=add_self_loops,
                    normalize=self.normalize,
                    bias=self.bias
                )
            )
            self.gcn_conv[i].reset_parameters()

        self.name: str = (
            f"GCNconv: node_features={transform_mode}, add_relu={self.add_relu} , bias={self.bias}, dropout={self.dropout}"
        )

    ############################
    def reset_parameters(self):
        """
        performs weight initialization
        :return:
        """
        if self.init == 'none':
            for i in range(len(self.gcn_conv)):
                self.gcn_conv[i].reset_parameters()

        if self.init == "xavier":
            for i in range(len(self.gcn_conv)):
                torch.nn.init.xavier_uniform_(self.gcn_conv[i].weight)

        elif self.init == "orthogonal":
            for i in range(len(self.gcn_conv)):
                torch.nn.init.xavier_uniform_(self.gcn_conv[i].weight)

        if self.bias:
            for i in range(len(self.gcn_conv)):
                torch.nn.init.zeros_(self.gcn_conv[i].bias)

    ############################
    def forward(self, graph:Data) -> torch.tensor:
        """
        :param x:
        :param edge_index:
        :return:
        """

        x, edge_index = graph.x, graph.edge_index

        for i, layer in enumerate(self.gcn_conv):
            x = layer(x, edge_index)

            if self.add_relu and i < (len(self.gcn_conv) - 1):
                x = F.relu(x)

            if self.dropout and i < (len(self.gcn_conv) - 1):
                x = F.dropout(x, p=self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)

    ############################
    def __repr__(self) -> str:

        h_dims = ''
        for i, dim in enumerate(self.hidden_dim):
            if i < len(self.hidden_dims)-1:
                h_dims += f'{dim} -> '
            else:
                h_dims += f'{dim}'

        nm = self.__class__.__name__
        return(f'{nm}: ({self.in_dim} -> {h_dims} -> {self.out_dim})')


#########################################################################################################################
class DynamicMLP(Module):

    def __init__(
            self,
            in_dim: int,
            hidden_dims: Union[tuple[int, int], list[int]],
            out_dim: int,
            add_relu: bool = True,
            bias: bool = True,
            init: str = "xavier",
            **kwargs
    ):
        """

        :param in_dim: input dimension
        :param hidden_dim: if is a list, contains the list of each hidden layer (len shows the number of hidden layers)
                           if is an integer, it shows the number of hidden layers
        :param out_dim: output dimension
        :param num_lin: number of linear layer, this value is ignored if hidden_dim is a list
        :param add_relu: Boolean, if True add relu after each linear, default is True
        :param bias: Boolean, if True adds bias to linear layer, default is True
        :param init: a string, identify the ways of weight initialization, either "xavier" or "orthogonal"
        """
        super().__init__()

        if isinstance(hidden_dims, tuple):
            hidden_dim = [hidden_dims[1]] * hidden_dims[0]

        dims = [in_dim] + hidden_dims + [out_dim]

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.add_relu = add_relu
        self.bias = bias
        self.init = init
        self.dims = dims
        self.dropout = kwargs['drop_out'] if 'drop_out' in kwargs else None

        ############################
        print(f"dims are {dims}")

        self.linears = ModuleList()
        for i in range(len(self.dims) - 1):
            self.linears.append(Linear(dims[i], dims[i + 1], bias=self.bias))
        self.reset_parameters()

    ############################
    def reset_parameters(self) -> None:
        """
        performs weight initialization
        :return:
        """
        if self.init == "xavier":
            for i in range(0, len(self.linears)):
                torch.nn.init.xavier_uniform_(self.linears[i].weight)

        elif self.init == "orthogonal":
            for i in range(len(self.linears)):
                torch.nn.init.xavier_uniform_(self.linears[i].weight)

        if self.bias:
            for i in range(len(self.linears)):
                torch.nn.init.zeros_(self.linears[i].bias)

    ############################ forward
    def forward(self, graph) -> torch.tensor:
        """

        :param x: feature map (node features
        :return:
        """
        x = graph.x
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if self.add_relu and i < (len(self.linears) - 1):
                x = F.relu(x)
            if self.dropout and i < (len(self.linears) - 1):
                x = F.dropout(x, p=self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)

    ############################
    def __repr__(self) -> str:

        hdims = ''
        for i, dim in enumerate(self.hidden_dim):
            if i < len(self.hidden_dims) - 1:
                hdims += f'{dim} -> '
            else:
                hdims += f'{dim}'

        nm = self.__class__.__name__
        return (f'{nm}: ({self.in_dim} -> {hdims} -> {self.out_dim} )')