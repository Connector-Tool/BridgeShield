import torch.nn
from torch_geometric.nn import HANConv, Linear, MLP, global_mean_pool, BatchNorm, global_max_pool

device = "cuda" if torch.cuda.is_available() else "cpu"


class SelfAttentionPooling(torch.nn.Module):
    def __init__(self, in_dim):
        super(SelfAttentionPooling, self).__init__()
        self.attn_fc = torch.nn.Linear(in_dim, 1)

    def forward(self, x, batch):
        attn_scores = self.attn_fc(x)
        attn_scores = torch.sigmoid(attn_scores)

        weighted_x = attn_scores * x
        graph_embedding = torch.zeros(batch.max() + 1, x.size(1)).to(x.device)
        graph_embedding = graph_embedding.scatter_add(0, batch.unsqueeze(-1).expand(-1, x.size(1)), weighted_x)

        return graph_embedding


class BridgeShield(torch.nn.Module):
    def __init__(
            self,
            out_channels: int,
            metadata: tuple,
            hidden_channels: int,
            num_layers: int,
            num_heads: int,
            graph_pooling: bool = True
    ):

  
        super().__init__()

        self.node_type2index = {t: i for i, t in enumerate(metadata[0])}
        self.edge_type2index = {t: i for i, t in enumerate(metadata[1])}

        self.mlps = torch.nn.ModuleDict()
        for node_type in self.node_type2index.keys():
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
            conv = HANConv(
                hidden_channels, hidden_channels, heads=num_heads,
                dropout=0.6, metadata=metadata
            )
            self.convs.append(conv)
            self.bns.append(BatchNorm(in_channels=hidden_channels))

        self.lin = Linear(hidden_channels, out_channels)
        self.graph_pooling = graph_pooling
        

    def forward(self, data):
        node_types, edge_types = data.metadata()
        x_dict = {
            t: torch.as_tensor(data[t].x, dtype=torch.float).to(device) if data[t].x is not None else torch.zeros(
                (data[t].num_nodes, 64)).to(device)
            for t in node_types
        }
        edge_index_dict = {t: torch.as_tensor(data[t].edge_index, dtype=torch.long).to(device) for t in edge_types}

        for node_type, x in x_dict.items():
            if x is None:
                x = torch.zeros((data[node_type].num_nodes, 64)).to(device)
            x_dict[node_type] = self.mlps[node_type](x).relu_()

        for conv in self.convs:
            try:
                x_dict = conv(x_dict, edge_index_dict)
            except Exception as e:
                print(e)

        if self.graph_pooling:
            attn_pooling = SelfAttentionPooling(in_dim=64).to(device)

            x_list, batch_list = [], []
            for t in node_types:
                x = x_dict.get(t)
                if x is None:
                    feature_dim = 64
                    num_nodes = data[t].num_nodes
                    x = torch.zeros((num_nodes, feature_dim)).to(device)
                x_list.append(x)
                batch_list.append(data[t].batch.to(device))

            if x_list:
                x = torch.cat(x_list, dim=0)
                batch = torch.cat(batch_list, dim=0)
                # x = attn_pooling(x, batch)
                # x = global_mean_pool(x, batch)
                x = global_max_pool(x, batch)
            else:
                x = torch.zeros((1, 64)).to(device)

            x = self.lin(x)
            return x


