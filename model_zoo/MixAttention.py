import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_adj


class MixAttention(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(MixAttention, self).__init__()
        self.dropout = dropout
        self.out_features = out_features
        self.Ws = Linear(in_features, out_features, weight_initializer='glorot')
        self.Wc = Linear(in_features, out_features, weight_initializer='glorot')

        self.a_context = Linear(2*out_features, 1, bias=False,
                        weight_initializer='glorot')
        self.a_structure = Linear(2*out_features, 1, bias=False,
                        weight_initializer='glorot')
        self.Ws_coff = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.Ws_coff, gain=1.414)
        self.Wc_coff = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.Wc_coff, gain=1.414)

    def forward(self, h_context, h_structure, edge_index: Adj):
        h_context = self.Wc(h_context)
        num_nodes = h_structure.size(0)
        context_input = torch.cat([h_context.repeat(1, num_nodes).view(num_nodes * num_nodes, -1),
                             h_context.repeat(num_nodes, 1)], dim=1).view(num_nodes, -1, 2 * self.out_features)
        context_alpha = F.leaky_relu(self.a_context(context_input).squeeze(2))

        h_structure = F.softmax(h_structure, dim=1)
        h_structure = self.Ws(h_structure)
        structure_input = torch.cat([h_structure.repeat(1, num_nodes).view(num_nodes * num_nodes, -1),
                             h_structure.repeat(num_nodes, 1)], dim=1).view(num_nodes, -1, 2 * self.out_features)
        structure_alpha = F.leaky_relu(self.a_structure(structure_input).squeeze(2))
        alpha = abs(self.Ws_coff) * context_alpha + abs(self.Wc_coff) * structure_alpha

        zero_vec = -9e15 * torch.ones_like(alpha)
        adj = to_dense_adj(edge_index).squeeze(0)

        attention = torch.where(adj>0, alpha, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        out = torch.matmul(attention, h_context)
        return out