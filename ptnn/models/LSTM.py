import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(Actor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.act1 = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)

        self.mlp_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        ])

        self.skip_connection = nn.Linear(input_size, hidden_size)
        self.act2 = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)

        h0 = nn.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
        c0 = nn.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
        nn.init.xavier_normal_(h0)
        nn.init.xavier_normal_(c0)

        packed_output, _ = self.lstm(packed_input, (h0, c0))
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        outputs = self.act1(outputs)
        outputs = self.layer_norm(outputs)
        outputs = self.dropout_layer(outputs)

        x = self.skip_connection(x)
        x = self.act2(x)
        outputs = outputs + x

        outputs = torch.mean(outputs, dim=1)
        for layer in self.mlp_layers:
            outputs = layer(outputs)

        return outputs


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.act1 = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)

        self.mlp_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        ])

        self.skip_connection = nn.Linear(input_size, hidden_size)
        self.act2 = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)

        h0 = nn.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
        c0 = nn.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
        nn.init.xavier_normal_(h0)
        nn.init.xavier_normal_(c0)

        packed_output, _ = self.lstm(packed_input, (h0, c0))
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        outputs = self.act1(outputs)
        outputs = self.layer_norm(outputs)
        outputs = self.dropout_layer(outputs)

        x = self.skip_connection(x)
        x = self.act2(x)
        outputs = outputs + x

        outputs = torch.mean(outputs, dim=1)
        for layer in self.mlp_layers:
            outputs = layer(outputs)

        return outputs


class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size,  num_layers, action_size):
        super(ActorCritic, self).__init__()
        self.actor = Actor(input_size, hidden_size, num_layers, action_size)
        self.critic = Critic(input_size, hidden_size, num_layers)

    def forward(self, state, seq_lengths):
        action_probs = self.actor(state, seq_lengths)
        value = self.critic(state, seq_lengths)
        return action_probs, value
