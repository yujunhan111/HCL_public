import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn_utils

class RNNEncoder(nn.Module):
    """
    对任意 [batch_size, seq_len, input_dim] 的序列做 RNN 聚合，
    输出每个样本的最后时刻向量 [batch_size, hidden_size]。
    支持 mask: bool类型 [batch_size, seq_len]。
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        rnn_cls = getattr(nn, rnn_type)  # RNN, GRU, or LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            # 下投影回 hidden_size
            self.down_projection = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.down_projection = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, input_size], float
        mask: [batch_size, seq_len], bool
        返回: [batch_size, hidden_size]
        """
        # 无效位置置0
        x = x * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp(min=1)  # 避免出现0

        # pack
        packed = rnn_utils.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.rnn(packed)
        # 解包
        outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)

        if self.rnn_type == "LSTM":
            # hidden = (h_n, c_n), 取最后 h_n
            last_h = hidden[0]
        else:
            last_h = hidden  # [num_layers * num_directions, batch_size, hidden_size]

        # 取最后一层
        # shape = [num_layers * num_directions, batch_size, hidden_size]
        if self.bidirectional:
            # shape => [num_layers, 2, batch_size, hidden_size]
            last_h = last_h.view(self.num_layers, 2, -1, self.hidden_size)
            # 取最后一层 (index = num_layers-1) => [2, batch_size, hidden_size]
            last_layer = last_h[self.num_layers - 1]
            # 拼接 forward + backward
            last_layer_cat = torch.cat((last_layer[0], last_layer[1]), dim=-1)  # [batch_size, hidden_size*2]
            out = self.down_projection(last_layer_cat)
        else:
            # shape=[num_layers, batch_size, hidden_size]
            last_layer = last_h[self.num_layers - 1]  # [batch_size, hidden_size]
            out = last_layer

        out = self.dropout(out)
        return out
