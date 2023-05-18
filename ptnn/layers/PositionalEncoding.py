import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Create the positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model).to("cuda")

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with added positional encodings
        """
        concatenated_array = torch.cat((self.pe[2][0], self.pe[1][0], self.pe[0][0]), axis=0)
        x = x + concatenated_array
        return x
