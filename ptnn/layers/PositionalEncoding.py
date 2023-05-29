import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        position = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-torch.log(torch.tensor(10000.0)) / self.d_model))
        pos_enc = torch.zeros(self.max_seq_len, self.d_model).to('cuda')

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.position_weights = pos_enc.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with added positional encodings
        """
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError("Input sequence length exceeds the maximum sequence length for positional encoding.")

        pos_enc = self.position_weights[:, :seq_len, :]
        x = x + pos_enc

        return x
