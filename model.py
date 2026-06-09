# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Small helpers

def _scatter_add(dst_index: torch.Tensor, src_val: torch.Tensor, N: int) -> torch.Tensor:
    out = torch.zeros((N, src_val.size(1)), device=src_val.device, dtype=src_val.dtype)
    out.index_add_(0, dst_index, src_val)
    return out

# GGNN (weighted)

class GGNNEncoderWeighted(nn.Module):
    def __init__(self, num_nodes: int, node_feat_dim: int, hidden_dim: int, layers: int = 3,
                 use_edge_weight: bool = True, dropout: float = 0.0):
        super().__init__()
        self.layers = int(layers)
        self.use_edge_weight = bool(use_edge_weight)
        self.dropout = float(dropout)

        self.in_lin = nn.Linear(node_feat_dim, hidden_dim)
        self.msg_lin = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.layers)])
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.layers)])

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor, edge_w: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = int(node_feat.size(0))
        src = edge_index[0]
        dst = edge_index[1]

        h = self.in_lin(node_feat)
        h = F.relu(h)

        for k in range(self.layers):
            m = h[src]
            m = self.msg_lin[k](m)
            if self.use_edge_weight and edge_w is not None:
                m = m * edge_w.view(-1, 1)
            m_agg = _scatter_add(dst, m, N)
            if self.dropout > 0:
                m_agg = F.dropout(m_agg, p=self.dropout, training=self.training)
            h = self.gru(m_agg, h)
            h = self.norm[k](h)
            h = F.relu(h)

        return h


# EdgeCostModel

class EdgeCostModel(nn.Module):
    def __init__(self, num_nodes, node_feat_dim, edge_feat_dim,
                 hidden_dim=64, gnn_layers=3, backbone="ggnn",
                 use_edge_weight=True, dropout=0.1, gat_heads=2):
        super().__init__()

        self.gnn = GGNNEncoderWeighted(
            num_nodes=num_nodes,
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            layers=gnn_layers,
            use_edge_weight=use_edge_weight,
            dropout=0.0,
        )
        self.use_edge_weight = bool(use_edge_weight)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, node_feat, edge_index, edge_w, edge_u, edge_v, edge_feat):
        h = self.gnn(node_feat, edge_index, edge_w=edge_w)
        z = torch.cat([h[edge_u], h[edge_v], edge_feat], dim=1)
        out = self.head(z)
        return F.softplus(out)