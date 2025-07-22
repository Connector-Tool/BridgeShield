import torch.nn
from torch_geometric.nn import HGTConv, Linear, MLP, global_mean_pool, BatchNorm, global_max_pool

device = "cuda" if torch.cuda.is_available() else "cpu"

class HGT(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            out_channels: int,
            metadata: tuple,
            num_layers: int = 2,
            num_heads: int = 4,
    ):
        super().__init__()

        self.mlps = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.mlps[node_type] = MLP(
                in_channels=-1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=2,
                norm=None,
            )

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(
                hidden_channels, hidden_channels,
                metadata, num_heads,
            ))
            self.bns.append(BatchNorm(in_channels=hidden_channels))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data, **kwargs):
        node_types, edge_types = data.metadata()
        x_dict = {t: torch.as_tensor(data[t].x, dtype=torch.float) for t in node_types}
        edge_index_dict = {t: torch.as_tensor(data[t].edge_index, dtype=torch.long) for t in edge_types}

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.mlps[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        x_list = [x_dict[t] for t in node_types]
        x = torch.cat(x_list, dim=0)
        batch = [data[t].batch for t in node_types]
        batch = torch.cat(batch, dim=0)
        # x = global_mean_pool(x, batch)
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x
