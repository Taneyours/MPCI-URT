import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import TransformerConv

class MPCIURT(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        gnn_hidden_1=1024,
        gnn_hidden_2=1024,
        lstm_hidden=256,
        lstm_layers=2,
        fusion_hidden=128,
    ):
        super().__init__()
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden

        self.conv1 = TransformerConv(input_dim, gnn_hidden_1)
        self.conv2 = TransformerConv(gnn_hidden_1, gnn_hidden_2)

        self.lstm = nn.LSTM(
            input_size=gnn_hidden_2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.destination_head = Linear(lstm_hidden, output_dim)
        self.fusion_1 = Linear(2 * output_dim, fusion_hidden)
        self.fusion_2 = Linear(fusion_hidden, output_dim)

    def forward(self, data, edge_index):
        x = data.x
        trajectory_scores = data.trajectory_scores

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        # Preserve the organization of the supplied experimental script.
        x = x.unsqueeze(1)

        h0 = torch.zeros(
            self.lstm_layers, x.size(0), self.lstm_hidden, device=x.device
        )
        c0 = torch.zeros(
            self.lstm_layers, x.size(0), self.lstm_hidden, device=x.device
        )

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        destination_logits = self.destination_head(x)

        fused = torch.cat((destination_logits, trajectory_scores), dim=1)
        fused = self.fusion_1(fused)
        fused = self.fusion_2(fused)
        return fused
