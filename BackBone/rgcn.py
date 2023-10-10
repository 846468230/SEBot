from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
import torch
import torch.nn as nn
from torch_geometric.utils import scatter
import torch.nn.functional as F


class RGCNConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations=2,
                 num_hop=2,
                 dropout=0.3):
        super(RGCNConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.dropout = dropout
        self.num_hop = num_hop
        self.root_weight = nn.Linear(in_channels, out_channels, bias=False)
        self.weight = Parameter(
            torch.Tensor(num_relations, in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
        self.hop_weight = Parameter(torch.Tensor(num_relations, num_hop))
        self.relation_weight = nn.Linear((num_relations + 1) * out_channels,
                                         (num_relations + 1) * out_channels)
        self.relu = nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.root_weight.weight)
        torch.nn.init.kaiming_normal_(self.weight)
        torch.nn.init.kaiming_normal_(self.relation_weight.weight)
        torch.nn.init.kaiming_normal_(self.hop_weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type):
        root_x = self.root_weight(x)

        x_relation = [root_x]
        for r in range(self.num_relations):
            x_r = torch.matmul(x, self.weight[r])
            x_hops = [x_r]
            hop_weight = torch.softmax(self.hop_weight[r],
                                       dim=0)  # shape: [num_hops]
            for hop in range(self.num_hop):
                x_hop = self.propagate(edge_index[:, edge_type == r],
                                       x=x_hops[-1],
                                       node_num=x.size(0))
                x_hops.append(x_hop)

            for hop in range(self.num_hop):
                x_hops[hop + 1] = x_hops[hop + 1] * hop_weight[hop]

            x_r = torch.sum(torch.stack(x_hops[1:]), dim=0)
            x_relation.append(x_r)
        total_x_relation = torch.cat(
            x_relation,
            dim=1)  # shape: [num_nodes, num_relations * out_channels]
        relation_weight = self.relation_weight(total_x_relation)
        relation_weight = self.relu(relation_weight)
        relation_weight = relation_weight.view(-1, self.num_relations + 1,
                                               self.out_channels)
        relation_weight = torch.softmax(
            relation_weight,
            dim=1)  # shape: [num_nodes, num_relations, out_channels]
        total_x_relation = total_x_relation.view(-1, self.num_relations + 1,
                                                 self.out_channels)

        x = torch.sum(total_x_relation * relation_weight, dim=1)

        return x

    def message(self, x_i: Tensor, x_j: Tensor, edge_index: Tensor,
                node_num: int) -> Tensor:
        index = edge_index[1]
        src = torch.ones_like(index).float().to(x_j.device)
        norm = scatter(src, index, dim=0, dim_size=node_num,
                       reduce='sum').clamp_(1.0)
        x_j = x_j / norm.index_select(0, index).unsqueeze(-1)
        return x_j


class FACNConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations=2,
                 num_heads=2,
                 temperature=0.01,
                 rcm=True,
                 dropout=0.3):
        super(FACNConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.dropout = dropout
        self.temperature = temperature
        self.rcm = rcm
        self.root_weight = nn.Linear(in_channels, out_channels)
        self.weight = Parameter(
            torch.Tensor(num_relations, in_channels, out_channels))
        self.relation_weight = nn.Linear(num_relations * out_channels,
                                         num_relations * out_channels)
        self.att = Parameter(torch.Tensor(num_relations, out_channels * 2, 1))
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.attention = []

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.root_weight.weight)
        torch.nn.init.kaiming_uniform_(self.weight)
        torch.nn.init.kaiming_uniform_(self.relation_weight.weight)
        torch.nn.init.kaiming_uniform_(self.att)
        torch.nn.init.zeros_(self.root_weight.bias)
        torch.nn.init.zeros_(self.relation_weight.bias)

    def forward(self, x, edge_index, edge_type, return_attention=False):
        root_x = self.root_weight(x)

        x_relation = []
        for r in range(self.num_relations):
            x_r = torch.matmul(x, self.weight[r])
            x_r = self.propagate(edge_index[:, edge_type == r],
                                 x=x_r,
                                 r=r,
                                 node_num=x.size(0),
                                 return_attention=return_attention)
            x_relation.append(x_r)
        if self.rcm:
            total_x_relation = torch.cat(
                x_relation,
                dim=1)  # shape: [num_nodes, num_relations * out_channels]
            relation_weight = self.relation_weight(total_x_relation)
            relation_weight = self.relu(relation_weight)
            relation_weight = relation_weight.view(-1, self.num_relations,
                                                   self.out_channels)
            relation_weight = torch.softmax(
                relation_weight,
                dim=1)  # shape: [num_nodes, num_relations, out_channels]
            total_x_relation = total_x_relation.view(-1, self.num_relations,
                                                     self.out_channels)

            x = torch.sum(total_x_relation * relation_weight, dim=1) + root_x
        else:
            x = torch.sum(torch.stack(x_relation), dim=0) + root_x
        if return_attention:
            torch.save(
                torch.cat(self.attention, dim=0).cpu().detach(),
                'edge_weight.pt')
        return x

    def message(self, x_i: Tensor, x_j: Tensor, edge_index: Tensor, r: int,
                node_num: int, return_attention: bool) -> Tensor:
        index = edge_index[1]
        src = torch.ones_like(index).float().to(x_j.device)
        norm = scatter(src, index, dim=0, dim_size=node_num,
                       reduce='sum').clamp_(1.0)
        att = torch.cat([x_i, x_j], dim=-1)
        att = torch.matmul(att, self.att[r])

        eps = torch.rand(1).to(x_j.device)
        att = att + torch.log(eps / (1 - eps))
        att = self.tanh(att / self.temperature)
        if return_attention:
            self.attention.append(att.squeeze(-1))
        att = F.dropout(att, p=self.dropout, training=self.training)
        x_j = x_j / norm.index_select(0, index).unsqueeze(-1) * att
        return x_j


if __name__ == '__main__':
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
    edge_type = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    conv = RGCNConv(16, 32, 2, 2)
    out = conv(x, edge_index, edge_type)
    print(out.shape)
