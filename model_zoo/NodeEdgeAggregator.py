import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from model_zoo.timespan_model import TSAEncoder, TSAEncoderV2
from model_zoo import EdgeToNodeConv
from torch_geometric.nn import Linear
from model_zoo.baseline.sage import SAGE
from model_zoo.MixAttention import MixAttention


class EdgeGCN(nn.Module):
    '''
        Edge GCN with TGAT as temporal edge attribute encoder
    '''
    def __init__(self, num_features, hidden_size, out_size, time_encoder = None):
        super(EdgeGCN, self).__init__()
        self.tsa_encoder = TSAEncoder(num_features, hidden_size, hidden_size, time_encoder)
        self.etn_conv = EdgeToNodeConv(hidden_size, hidden_size)
        self.linear = Linear(hidden_size, out_size, bias=False, weight_initializer='glorot')

    def forward(self, x, H, edge_index):
        tsae_out = self.tsa_encoder(x, edge_index)
        out = self.etn_conv(tsae_out, H)
        out = F.leaky_relu(out, negative_slope=0.2)
        return self.linear(out)


class EdgeGCNV2(nn.Module):
    '''
        Edge GCN with GAT as non-temporal edge attribute encoder
    '''
    def __init__(self, num_features, hidden_size, out_size):
        super(EdgeGCNV2, self).__init__()
        self.gat = GATConv(num_features, hidden_size)
        self.etn_conv = EdgeToNodeConv(hidden_size, hidden_size)
        self.linear = Linear(hidden_size, out_size, bias=False, weight_initializer='glorot')

    def forward(self, x, H, edge_index):
        gat_out = self.gat(x, edge_index)
        out = self.etn_conv(gat_out, H)
        out = F.leaky_relu(out, negative_slope=0.2)
        return self.linear(out)


class EdgeGCNV3(nn.Module):
    '''
        Edge GCN with GAT as non-temporal edge attribute encoder
    '''
    def __init__(self, et_size, ea_size, hidden_size, out_size, time_encoder = None):
        super(EdgeGCNV3, self).__init__()
        self.tsa_encoder = TSAEncoderV2(et_size, ea_size, hidden_size, hidden_size, time_encoder)
        self.etn_conv = EdgeToNodeConv(hidden_size, hidden_size)
        self.linear = Linear(hidden_size, out_size, bias=False, weight_initializer='glorot')

    def forward(self, et, ea, H, edge_index):
        tsae_out = self.tsa_encoder(et, ea, edge_index)
        out = self.etn_conv(tsae_out, H)
        out = F.leaky_relu(out, negative_slope=0.2)
        return self.linear(out)


class NodeEdgeAggregator(nn.Module):
    def __init__(self, args):
        super(NodeEdgeAggregator, self).__init__()
        self.edge_model = EdgeGCN(args.timestamp_size, args.hidden_size, args.hidden_size, args.time_encoder)
        self.node_model = SAGE(args.num_features, args.hidden_size, args.hidden_size, args.num_layers, args.dropout, False)
        self.linear = Linear(2*args.hidden_size, args.out_size, bias=False, weight_initializer='glorot')

    def forward(self, x, et, H, raw_edge_index, lg_edge_index):
        edge_repr = self.edge_model(et, H, lg_edge_index)
        node_repr = self.node_model(x, raw_edge_index)
        out = torch.cat([edge_repr, node_repr], dim=1)
        return F.log_softmax(self.linear(out), dim=1)


class NodeEdgeAggregatorV2(nn.Module):
    def __init__(self, args):
        super(NodeEdgeAggregatorV2, self).__init__()
        self.edge_model = EdgeGCN(args.timestamp_size, args.hidden_size, args.hidden_size, args.time_encoder)
        self.edge_aggr = SAGE(args.hidden_size, args.hidden_size, args.hidden_size, args.num_layers-1,
                                    args.dropout, False)
        self.attr_node_model = SAGE(args.num_features, args.hidden_size, args.hidden_size, args.num_layers,
                                    args.dropout, False)

        self.mix_attention = MixAttention(args.hidden_size, args.hidden_size, dropout=args.dropout)
        self.linear = Linear(args.hidden_size, args.out_size, bias=False, weight_initializer='glorot')

    def forward(self, x, et, H, raw_edge_index, lg_edge_index):
        edge_repr = self.edge_model(et, H, lg_edge_index)
        aggr_edge_repr = self.edge_aggr(edge_repr, raw_edge_index)
        node_repr = self.attr_node_model(x, raw_edge_index)
        out = self.mix_attention(node_repr, aggr_edge_repr, raw_edge_index)
        return F.log_softmax(self.linear(out), dim=1)

    def embeds(self, x, et, H, raw_edge_index, lg_edge_index):
        edge_repr = self.edge_model(et, H, lg_edge_index)
        aggr_edge_repr = self.edge_aggr(edge_repr, raw_edge_index)
        node_repr = self.attr_node_model(x, raw_edge_index)
        out = self.mix_attention(node_repr, aggr_edge_repr, raw_edge_index)
        return out


class NodeEdgeAggregatorV2WithoutEdgeAggr(nn.Module):
    def __init__(self, args):
        super(NodeEdgeAggregatorV2WithoutEdgeAggr, self).__init__()
        self.edge_model = EdgeGCN(args.timestamp_size, args.hidden_size, args.hidden_size, args.time_encoder)

        self.attr_node_model = SAGE(args.num_features, args.hidden_size, args.hidden_size, args.num_layers,
                                    args.dropout, False)

        self.mix_attention = MixAttention(args.hidden_size, args.hidden_size, dropout=args.dropout)
        self.linear = Linear(args.hidden_size, args.out_size, bias=False, weight_initializer='glorot')

    def forward(self, x, et, H, raw_edge_index, lg_edge_index):
        edge_repr = self.edge_model(et, H, lg_edge_index)
        node_repr = self.attr_node_model(x, raw_edge_index)
        out = self.mix_attention(node_repr, edge_repr, raw_edge_index)
        return F.log_softmax(self.linear(out), dim=1)


class NodeEdgeAggregatorV3(nn.Module):
    def __init__(self, args):
        super(NodeEdgeAggregatorV3, self).__init__()
        self.edge_model = EdgeGCNV2(args.edge_attr_size, args.hidden_size, args.hidden_size)
        self.edge_aggr = SAGE(args.hidden_size, args.hidden_size, args.hidden_size, args.num_layers - 1,
                              args.dropout, False)
        self.attr_node_model = SAGE(args.num_features, args.hidden_size, args.hidden_size, args.num_layers,
                                    args.dropout, False)

        self.mix_attention = MixAttention(args.hidden_size, args.hidden_size, dropout=args.dropout)
        self.linear = Linear(args.hidden_size, args.out_size, bias=False, weight_initializer='glorot')

    def forward(self, x, et, H, raw_edge_index, lg_edge_index):
        edge_repr = self.edge_model(et, H, lg_edge_index)
        aggr_edge_repr = self.edge_aggr(edge_repr, raw_edge_index)
        node_repr = self.attr_node_model(x, raw_edge_index)
        out = self.mix_attention(node_repr, aggr_edge_repr, raw_edge_index)
        out = self.linear(out)
        return F.log_softmax(out, dim=1)


class NodeEdgeAggregatorV4(nn.Module):
    def __init__(self, args):
        super(NodeEdgeAggregatorV4, self).__init__()
        self.edge_model = EdgeGCNV3(args.timestamp_size, args.edge_attr_size, args.hidden_size, args.hidden_size, args.time_encoder)
        self.edge_aggr = SAGE(args.hidden_size, args.hidden_size, args.hidden_size, args.num_layers-1,
                                    args.dropout, False)
        self.attr_node_model = SAGE(args.num_features, args.hidden_size, args.hidden_size, args.num_layers,
                                    args.dropout, False)

        self.mix_attention = MixAttention(args.hidden_size, args.hidden_size, dropout=args.dropout)
        self.linear = Linear(args.hidden_size, args.out_size, bias=False, weight_initializer='glorot')

    def forward(self, x, et, ea, H, raw_edge_index, lg_edge_index):
        edge_repr = self.edge_model(et, ea, H, lg_edge_index)
        aggr_edge_repr = self.edge_aggr(edge_repr, raw_edge_index)
        node_repr = self.attr_node_model(x, raw_edge_index)
        out = self.mix_attention(node_repr, aggr_edge_repr, raw_edge_index)
        return F.log_softmax(self.linear(out), dim=1)