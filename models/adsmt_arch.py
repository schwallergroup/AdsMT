import os
import errno
import logging
import torch
from torch import nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_batch
from ocpmodels.common.registry import registry
from ocpmodels.models.scn.smearing import GaussianSmearing
from ocpmodels.common.utils import conditional_grad, _report_incompat_keys

from models.encoders import *
from models.freeze_layers import frozen_layer_name, conv_name


def atom_height(data):
    z_norm_vecs = torch.tensor([], device=data.pos.device)
    for cell in data.cell:
        a = cell[0]
        b = cell[1]
        z_vec = torch.cross(a, b)
        z_norm_vec = F.normalize(z_vec, dim=-1).unsqueeze(0)
        z_norm_vecs = torch.cat((z_norm_vecs, z_norm_vec), 0)
    surface_batch = data.batch
    z_norm_vecs = z_norm_vecs[surface_batch].unsqueeze(1)
    pos = data.pos.unsqueeze(2)
    z_atoms = torch.bmm(z_norm_vecs, pos)
    z_atoms = torch.squeeze(z_atoms)
    z_max = scatter(z_atoms, surface_batch, dim_size=len(data), reduce='max')
    z_min = scatter(z_atoms, surface_batch, dim_size=len(data), reduce='min')
    z_max = z_max[surface_batch]
    z_min = z_min[surface_batch]
    h = (z_atoms - z_min) / (z_max - z_min)

    assert not torch.isnan(h).any()
    assert not torch.isinf(h).any() 
    return h


def get_graph_encoder(name, kwargs):
    # assert name in [], f"Graph Encoder '{name}' is unavailable!"
    encoder = globals().get(name)
    if encoder:
        return encoder(**kwargs)
    else:
        raise ValueError(f"Graph Encoder '{name}' not found!")


class CrossModal(nn.Module):
    def __init__(
        self,
        vec_emb_dim: int = 128,
        node_emb_dim: int = 128,
        hidden_dim: int = 128,
        out_channels: int = 1,
        num_gaussians: int = 50,
        num_heads: int = 4,
        attn_layers: int = 1,
        mlp_layers: int = 3,
        dropout: float = 0,
        act: str = "silu",
        norm: str = None,
    ) -> None:

        super(CrossModal, self).__init__()
        assert vec_emb_dim == node_emb_dim
        self.attn_layers = attn_layers
        self.mlp_layers = mlp_layers

        self.node_pos_expansion = GaussianSmearing(
            start=0.0,
            stop=1.0,
            num_gaussians=num_gaussians,
        )
        self.fc_pos_exp = nn.Linear(num_gaussians, hidden_dim)

        # cross attention layers
        self.cross_attns = nn.ModuleList(
            [
                MultiheadAttention(
                    embed_dim=hidden_dim*2,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(attn_layers)
            ]
        )

        # self attention layers
        self.self_attns = nn.ModuleList(
            [
                MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(attn_layers)
            ]
        )

        self.mlp = MLP(
            in_channels=hidden_dim*4,
            hidden_channels=hidden_dim*2,
            out_channels=out_channels,
            num_layers=mlp_layers,
            dropout=dropout,
            act=act,
            norm=norm,
        )

    def forward(self, vec_emb, node_emb, surface, need_weights=False):

        # atom positional encodering along z direction
        batch = surface.batch
        atom_h = atom_height(surface)
        node_pos_emb = self.node_pos_expansion(atom_h)
        node_pos_emb = self.fc_pos_exp(node_pos_emb)

        # cross attention layers
        graph_emb = scatter(node_emb, batch, dim=0, reduce='mean')
        query = torch.cat([vec_emb, graph_emb], dim=1).unsqueeze(1)
        key = torch.cat([node_emb, node_pos_emb], dim=1)
        k_dense, mask = to_dense_batch(key, batch)
        value = torch.cat([node_emb, graph_emb[batch]], dim=1)
        v_dense, _ = to_dense_batch(value, batch)

        cross_weights = []
        for cross_attn in self.cross_attns:
            cross_out = cross_attn(query, k_dense, v_dense, key_padding_mask=~mask,
                                   need_weights=need_weights)
            query = query + cross_out[0]
            cross_weights.append(
                self.get_attn_weights(cross_out, mask, batch))
        h_attn1 = query.squeeze(1)

        # self attention layers
        ads_emb = torch.cat(
            [vec_emb.unsqueeze(1), graph_emb.unsqueeze(1)], dim=1
        )
        _mask = torch.ones(ads_emb.shape[:2]).bool().to(batch.device)
        atom_top_emb = node_emb[atom_h > 0.8]
        atom_top_batch = batch[atom_h > 0.8]
        at_emb_dense, mask = to_dense_batch(atom_top_emb, atom_top_batch)
        h = torch.cat([ads_emb, at_emb_dense], dim=1)
        mask = torch.cat([_mask, mask], dim=1)

        self_weights = []
        for self_attn in self.self_attns:
            self_out = self_attn(h, h, h, key_padding_mask=~mask,
                                 need_weights=need_weights)
            h = h + self_out[0]
            # self_weights.append(self_out[1])
        h_attn2 = torch.cat([h[:, 0, :], h[:, 1, :],], dim=1)

        h_attn = torch.cat([h_attn1, h_attn2], dim=1)
        energy = self.mlp(h_attn).view(-1)

        if need_weights:
            return energy, cross_weights, self_weights
        else:
            return energy

    def get_attn_weights(self, attn_out, mask, batch):
        out = []
        if attn_out[1] is None:
            return out

        attn_out = attn_out[1].squeeze(1)[mask]
        for i in range(mask.size(0)):
            out.append(attn_out[batch == i])
        return out


@registry.register_model("adsmt_arch")
class AdsMT_ARCH(nn.Module):
    """pyg implementation."""

    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        use_pbc: bool = True,
        otf_graph: bool = True,
        regress_forces: bool = False,
        pretrain: bool = False,
        ckpt_path: str = None,
        freeze_nblock: int = 0,
        graph_encoder: str = 'adsgt',
        graph_encoder_args: dict = None,
        desc_layers: int = 1,
        desc_hidden_dim: int = 128,
        cross_modal_args: dict = None,
    ):
        super().__init__()

        self.use_pbc = True
        self.otf_graph = True
        self.regress_forces = False

        # graph encoder for surface graph
        self.graph_encoder = get_graph_encoder(graph_encoder, graph_encoder_args)

        # vector encoder for adsorbate descriptors
        self.vector_encoder = MLP(in_channels=208, hidden_channels=desc_hidden_dim,
                                  out_channels=desc_hidden_dim, num_layers=desc_layers)

        # cross-modal encoder for fusing the embeddings of surface graph and adsorbate descriptors
        self.cross_encoder = CrossModal(**cross_modal_args)

        # load pretrained model from ckpt file
        if pretrain:
            assert ckpt_path is not None
            self.from_pretrain_ckpt(ckpt_path)
            # freeze some layer parameters
            if freeze_nblock > 0:
                self.freeze_layers(graph_encoder, freeze_nblock)

    @conditional_grad(torch.enable_grad())
    def forward(self, data, need_weights=False):

        node_emb = self.graph_encoder(data)
        vec_emb = self.vector_encoder(data.ads_des)

        if need_weights:
            energy, cross_weights, self_weights = self.cross_encoder(
                vec_emb, node_emb, data, need_weights)
            return energy, cross_weights, self_weights
        else:
            energy = self.cross_encoder(vec_emb, node_emb, data)
            return energy

    def from_pretrain_ckpt(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", ckpt_path
            )
        else:
            logging.info(f"Loading checkpoint from: {ckpt_path}")
            device = next(self.parameters()).device
            checkpoint = torch.load(ckpt_path, map_location=device)

        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
        mod_key_count = next(iter(self.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]

        incompat_keys = self.load_state_dict(new_dict, strict=False)
        return _report_incompat_keys(self, incompat_keys, strict=True)

    def freeze_layers(self, ge_name, nblock=0):
        frozen_layers = frozen_layer_name[ge_name]
        for name, child in self.graph_encoder.named_children():
            if name not in frozen_layers:
                continue
            for param in child.parameters():
                param.requires_grad = False

        for _name in conv_name[ge_name]:
            conv_layers = getattr(self.graph_encoder, _name)
            for i in range(nblock):
                for param in conv_layers[i].parameters():
                    param.requires_grad = False

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
