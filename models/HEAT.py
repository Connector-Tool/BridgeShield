import torch.nn
from torch_geometric.nn import HEATConv, MLP, global_mean_pool, BatchNorm, global_max_pool


class HEAT(torch.nn.Module):
    def __init__(
            self,
            out_channels: int,
            hidden_channels: int,
            num_layers: int,
            metadata: tuple,
            **kwargs
    ):
        super().__init__()

        self.node_lin = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.node_lin[node_type] = MLP(
                in_channels=-1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=2,
                norm=None,
            )

        self.edge_lin = torch.nn.ModuleDict()
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.edge_lin[edge_type] = MLP(
                in_channels=-1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=2,
                norm=None,
            )

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HEATConv(
                in_channels=-1,
                out_channels=hidden_channels,
                num_node_types=len(metadata[0]),
                num_edge_types=len(metadata[1]),
                edge_type_emb_dim=hidden_channels,
                edge_dim=hidden_channels,
                edge_attr_emb_dim=hidden_channels,
            ))
            self.bns.append(BatchNorm(in_channels=hidden_channels))

        self.out_lin = MLP(
            in_channels=-1,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=2,
            norm=None,
        )

    def forward(self, data, **kwargs):
        node_types, edge_types = data.metadata()
        for node_type in node_types:
            x = data[node_type].x
            data[node_type].x = self.node_lin[node_type](x.float())

        for edge_type in edge_types:
            edge_attr = data[edge_type].edge_attr
            data[edge_type].edge_attr = self.edge_lin['__'.join(edge_type)](edge_attr)

        data = data.to_homogeneous()
        for i, conv in enumerate(self.convs):
            data.x = conv(
                x=data.x,
                edge_index=data.edge_index,
                edge_type=data.edge_type,
                node_type=data.node_type,
                edge_attr=data.edge_attr,
            )
            data.x = self.bns[i](data.x)

        # emb = global_mean_pool(data.x, data.batch)
        emb = global_max_pool(data.x, data.batch)
        return self.out_lin(emb)
