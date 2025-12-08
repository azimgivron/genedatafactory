import torch
from torch import nn
from torch_geometric.nn import GCNConv


class FixedGCN(nn.Module):
    """Frozen GCN f0 that maps gene features and a graph to latent factors.

    Args:
        in_dim (int): Input feature dimension.
        hidden_dims (Iterable[int]): Sequence of hidden layer widths.
        out_dim (int): Output latent dimension.
    """

    def __init__(self, in_dim: int, hidden_dims, out_dim: int):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        self.convs = nn.ModuleList(
            [GCNConv(dims[l], dims[l + 1]) for l in range(len(dims) - 1)]
        )
        self.out_lin = nn.Linear(dims[-1], out_dim)
        self.activation = nn.ReLU()

        # Freeze parameters: they are sampled once, never trained
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute latent gene representation given node features and edges.

        Args:
            x (Tensor): Node feature matrix of shape (n_nodes, in_dim).
            edge_index (Tensor): Edge indices in COO format.

        Returns:
            Tensor: Shape (n_nodes, out_dim) containing latent factors.
        """
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.activation(h)
        out = self.out_lin(h)
        return out


class FixedMLP(nn.Module):
    """Frozen MLP g0 that maps disease features to latent factors.

    Args:
        in_dim (int): Input feature dimension.
        hidden_dims (Iterable[int]): Sequence of hidden layer widths.
        out_dim (int): Output latent dimension.
    """

    def __init__(self, in_dim: int, hidden_dims, out_dim: int):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for l in range(len(dims) - 2):
            layers.append(nn.Linear(dims[l], dims[l + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Compute latent disease representation from disease features.

        Args:
            y (Tensor): Disease feature matrix of shape (n_diseases, in_dim).

        Returns:
            Tensor: Shape (n_diseases, out_dim) containing latent factors.
        """
        return self.net(y)
