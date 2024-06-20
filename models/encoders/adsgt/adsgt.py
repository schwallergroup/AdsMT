import torch
from torch import nn
from ocpmodels.models.base import BaseModel
from ocpmodels.common.utils import conditional_grad
from ocpmodels.datasets.embeddings import KHOT_EMBEDDINGS
from models.encoders.adsgt.layers import AdsGTAttn, AtomPosEncoder, RBFExpansion


class AdsGT(BaseModel):
    """pyg implementation."""

    def __init__(
        self,
        conv_layers=5,
        node_features=128,
        edge_features=128,
        node_layer_head=4,
        cutoff=8.0,
        use_pbc=True,
        otf_graph=True,
        max_neighbors=12,
    ):
        super().__init__()

        self.cutoff = cutoff
        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.regress_forces = False

        embeddings = KHOT_EMBEDDINGS
        self.embedding = torch.zeros(100, len(embeddings[1]))
        for i in range(100):
            self.embedding[i] = torch.tensor(embeddings[i + 1])
        self.atom_embedding = nn.Linear(len(embeddings[1]), node_features)
        self.atomh_encoder = AtomPosEncoder(hidden_dim=node_features)

        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=cutoff,
                bins=edge_features,
            ),
            nn.Linear(edge_features, node_features),
            nn.Softplus(),
            nn.Linear(node_features, node_features),
        )

        self.att_layers = nn.ModuleList(
            [
                AdsGTAttn(
                    in_channels=node_features,
                    out_channels=node_features,
                    heads=node_layer_head,
                    edge_dim=node_features,
                ) for _ in range(conv_layers)
            ]
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):

        # Get node features
        if self.embedding.device != data.atomic_numbers.device:
            self.embedding = self.embedding.to(data.atomic_numbers.device)
        x = self.embedding[data.atomic_numbers.long() - 1]
        atom_h = self.atomh_encoder(data)
        node_features = self.atom_embedding(x)+ atom_h

        (
            edge_index,
            distances,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)
        edge_features = self.rbf(distances)

        for att_layer in self.att_layers:
            node_features = att_layer(node_features, edge_index, edge_features)

        return node_features
