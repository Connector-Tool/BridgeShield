import torch.nn
from torch_geometric.nn import RGCNConv, Linear, MLP, global_mean_pool, global_max_pool



device = "cuda" if torch.cuda.is_available() else "cpu"

class RGCN(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            out_channels: int,
            num_layers: int,
            metadata: tuple,
            graph_pooling: bool = True,
            **kwargs
    ):
        super().__init__()
        self.node_type2index = {t: i for i, t in enumerate(metadata[0])}
        self.edge_type2index = {t: i for i, t in enumerate(metadata[1])}

        self.node_lin = torch.nn.ModuleDict()
        for node_type in self.node_type2index.keys():
            self.node_lin[node_type] = MLP(
                in_channels=-1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=2,
                norm=None,
            )

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = RGCNConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                num_relations=41
            )
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        self.graph_pooling = graph_pooling

    def forward(self, data):
        node_types, edge_types = data.metadata()

        x = list()
        node_type_indices = list()
        for node_type in node_types:
            _x = data[node_type].x
            _x = _x.float()
            _x = self.node_lin[node_type](_x)
            x.append(_x)
            node_type_indices.extend([
                self.node_type2index[node_type]
                for _ in range(data[node_type].x.shape[0])
            ])
        x = torch.cat(x, dim=0)
        node_type_indices = torch.tensor(node_type_indices)

        edge_type_indices = list()
        for edge_type in edge_types:
            if not data[edge_type].edge_attr.shape[0] == 0:
                edge_type_indices.extend([
                    self.edge_type2index[edge_type]
                    for _ in range(data[edge_type].edge_attr.shape[0])
                ])
        edge_type_indices = torch.tensor(edge_type_indices)

        uid = 0
        node_type2index2uid = dict()
        for node_type in node_types:
            node_type2index2uid[node_type] = dict()
            for i in range(len(data[node_type].x)):
                node_type2index2uid[node_type][i] = uid
                uid += 1
        edge_indices = list()
        for edge_type in edge_types:
            from_node_type, _, to_node_type = edge_type
            edge_index = data[edge_type].edge_index.t().long().tolist()
            edge_index = [
                [
                    node_type2index2uid[from_node_type][edge_index[i][0]],
                    node_type2index2uid[to_node_type][edge_index[i][1]]
                ] for i in range(len(edge_index))
            ]
            edge_indices.extend(edge_index)
        edge_indices = torch.tensor(edge_indices).t().contiguous()
        edge_indices = edge_indices.long()

        x = x.to(device)
        edge_indices = edge_indices.to(device)
        edge_type_indices = edge_type_indices.to(device)

        for conv in self.convs:
            x = conv(
                x=x,
                edge_index=edge_indices,
                edge_type=edge_type_indices
            )

        if self.graph_pooling:
            batch = [data[t].batch for t in node_types]
            batch = torch.cat(batch, dim=0)
            # x = global_mean_pool(x, batch)
            x = global_max_pool(x, batch)
            x = self.lin(x)
            return x

