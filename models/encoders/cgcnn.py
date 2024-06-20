import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models.schnet import GaussianSmearing
from ocpmodels.datasets.embeddings import KHOT_EMBEDDINGS, QMOF_KHOT_EMBEDDINGS
from ocpmodels.models.base import BaseModel


class CGCNNEncoder(BaseModel):
    r"""Implementation of the Crystal Graph CNN model from the
    `"Crystal Graph Convolutional Neural Networks for an Accurate
    and Interpretable Prediction of Material Properties"
    <https://arxiv.org/abs/1710.10324>`_ paper.

    Args:
        atom_embedding_size (int, optional): Size of atom embeddings.
            (default: :obj:`64`)
        num_graph_conv_layers (int, optional): Number of graph convolutional layers.
            (default: :obj:`6`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        num_gaussians (int, optional): Number of Gaussians used for smearing.
            (default: :obj:`50.0`)
    """

    def __init__(
        self,
        atom_embedding_size: int = 64,
        num_graph_conv_layers: int = 6,
        cutoff: float = 6.0,
        num_gaussians: int = 50,
        embeddings: str = "khot",
    ) -> None:
        super(CGCNNEncoder, self).__init__()
        self.cutoff = cutoff
        self.use_pbc = True
        self.otf_graph = True
        self.regress_forces = False
        self.max_neighbors = 50
        # Get CGCNN atom embeddings
        if embeddings == "khot":
            embeddings = KHOT_EMBEDDINGS
        elif embeddings == "qmof":
            embeddings = QMOF_KHOT_EMBEDDINGS
        else:
            raise ValueError(
                'embedding mnust be either "khot" for original CGCNN K-hot elemental embeddings or "qmof" for QMOF K-hot elemental embeddings'
            )
        self.embedding = torch.zeros(100, len(embeddings[1]))
        for i in range(100):
            self.embedding[i] = torch.tensor(embeddings[i + 1])
        self.embedding_fc = nn.Linear(len(embeddings[1]), atom_embedding_size)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.convs = nn.ModuleList(
            [
                CGCNNConv(
                    node_dim=atom_embedding_size,
                    edge_dim=num_gaussians,
                    cutoff=cutoff,
                )
                for _ in range(num_graph_conv_layers)
            ]
        )

    def forward(self, data):
        # Get node features
        if self.embedding.device != data.atomic_numbers.device:
            self.embedding = self.embedding.to(data.atomic_numbers.device)
        data.x = self.embedding[data.atomic_numbers.long() - 1]

        (
            edge_index,
            distances,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        data.edge_index = edge_index
        data.edge_attr = self.distance_expansion(distances)
        # Forward pass through the network
        node_feats = self.embedding_fc(data.x)
        for f in self.convs:
            node_feats = f(node_feats, data.edge_index, data.edge_attr)

        return node_feats


class CGCNNConv(MessagePassing):
    """Implements the message passing layer from
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.
    """

    def __init__(
        self, node_dim, edge_dim, cutoff: float = 6.0, **kwargs
    ) -> None:
        super(CGCNNConv, self).__init__(aggr="add")
        self.node_feat_size = node_dim
        self.edge_feat_size = edge_dim
        self.cutoff = cutoff

        self.lin1 = nn.Linear(
            2 * self.node_feat_size + self.edge_feat_size,
            2 * self.node_feat_size,
        )
        self.bn1 = nn.BatchNorm1d(2 * self.node_feat_size)
        self.ln1 = nn.LayerNorm(self.node_feat_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.lin1.weight)

        self.lin1.bias.data.fill_(0)

        self.bn1.reset_parameters()
        self.ln1.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_feat_size]
        """
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )
        out = nn.Softplus()(self.ln1(out) + x)
        return out

    def message(self, x_i, x_j, edge_attr):
        """
        Arguments:
            x_i has shape [num_edges, node_feat_size]
            x_j has shape [num_edges, node_feat_size]
            edge_attr has shape [num_edges, edge_feat_size]

        Returns:
            tensor of shape [num_edges, node_feat_size]
        """
        z = self.lin1(torch.cat([x_i, x_j, edge_attr], dim=1))
        z = self.bn1(z)
        z1, z2 = z.chunk(2, dim=1)
        z1 = nn.Sigmoid()(z1)
        z2 = nn.Softplus()(z2)
        return z1 * z2
