import torch
import torch.nn as nn


class LSTMCritic(nn.Module):

    def __init__(
        self,
        state_dim=8,
        hidden_dim=128,
        fc_dim=64,
        num_agents=2
    ):

        super().__init__()

        self.num_agents = num_agents

        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.Tanh()
        )

        # One value for each agent
        self.value = nn.Linear(
            fc_dim,
            num_agents
        )

    def forward(self, state):

        # state:
        # [batch, sequence_length, state_dim]

        lstm_output, _ = self.lstm(state)

        # Take the final timestep
        last_output = lstm_output[:, -1, :]

        features = self.fc(last_output)

        # [batch, num_agents]
        values = self.value(features)

        return values
