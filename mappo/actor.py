import torch
import torch.nn as nn
from torch.distributions import Normal


class LSTMActor(nn.Module):

    def __init__(
        self,
        obs_dim=7,
        hidden_dim=128,
        fc_dim=64,
        action_dim=1,
        action_limit=3.0
    ):
        super().__init__()

        self.action_limit = action_limit

        # --------------------------------------------------
        # LSTM
        # --------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # --------------------------------------------------
        # Fully connected layer
        # --------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.Tanh()
        )

        # --------------------------------------------------
        # Mean of Gaussian policy
        # --------------------------------------------------
        self.mean = nn.Linear(
            fc_dim,
            action_dim
        )

        # --------------------------------------------------
        # Log standard deviation
        # --------------------------------------------------
        self.log_std = nn.Parameter(
            torch.zeros(action_dim)
        )

    def forward(self, observation):

        # observation shape:
        # [batch_size, sequence_length, 7]
        #
        # 7 features:
        #
        # 0 = own distance to conflict
        # 1 = own speed
        # 2 = other vehicle distance
        # 3 = other vehicle speed
        # 4 = own TTC
        # 5 = other vehicle TTC
        # 6 = other vehicle passed status

        lstm_output, _ = self.lstm(
            observation
        )

        # Take the output from the final timestep
        last_output = (
            lstm_output[:, -1, :]
        )

        # Fully connected representation
        features = self.fc(
            last_output
        )

        # Mean of Gaussian policy
        mean = self.mean(
            features
        )

        # Keep the Gaussian standard deviation positive
        std = torch.exp(
            self.log_std
        )

        return mean, std

    def get_action(
        self,
        observation
    ):

        mean, std = self.forward(
            observation
        )

        # Gaussian distribution
        distribution = Normal(
            mean,
            std
        )

        # Sample an action
        raw_action = distribution.rsample()

        # Squash action to [-1, +1]
        squashed_action = torch.tanh(
            raw_action
        )

        # Scale to environment action range [-3, +3]
        action = (
            self.action_limit *
            squashed_action
        )

        # --------------------------------------------------
        # Log probability correction for tanh transformation
        # --------------------------------------------------

        log_probability = (
            distribution.log_prob(
                raw_action
            )
        )

        correction = torch.log(
            1 -
            squashed_action.pow(2) +
            1e-6
        )

        log_probability = (
            log_probability -
            correction
        ).sum(
            dim=-1
        )

        # Entropy of original Gaussian distribution
        entropy = (
            distribution
            .entropy()
            .sum(dim=-1)
        )

        return (
            action,
            log_probability,
            entropy
        )

    def evaluate_action(
        self,
        observation,
        action
    ):

        mean, std = self.forward(
            observation
        )

        distribution = Normal(
            mean,
            std
        )

        # Convert bounded action [-3,+3]
        # back to [-1,+1]
        squashed_action = (
            action /
            self.action_limit
        )

        # Numerical safety
        squashed_action = torch.clamp(
            squashed_action,
            -0.999999,
            0.999999
        )

        # Inverse tanh
        raw_action = torch.atanh(
            squashed_action
        )

        # Log probability
        log_probability = (
            distribution.log_prob(
                raw_action
            )
        )

        correction = torch.log(
            1 -
            squashed_action.pow(2) +
            1e-6
        )

        log_probability = (
            log_probability -
            correction
        ).sum(
            dim=-1
        )

        entropy = (
            distribution
            .entropy()
            .sum(dim=-1)
        )

        return (
            log_probability,
            entropy
        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    # Create actor
    actor = LSTMActor(
        obs_dim=7,
        hidden_dim=128,
        fc_dim=64,
        action_dim=1,
        action_limit=3.0
    )

    # ------------------------------------------------------
    # Example input
    #
    # 4 sequences
    # 10 timesteps
    # 7 observation features
    # ------------------------------------------------------

    observation = torch.randn(
        4,
        10,
        7
    )

    # Get action
    action, log_probability, entropy = (
        actor.get_action(
            observation
        )
    )

    print(
        "Observation shape:",
        observation.shape
    )

    print(
        "Action shape:",
        action.shape
    )

    print(
        "Action:"
    )

    print(
        action
    )

    print(
        "Minimum action:",
        action.min().item()
    )

    print(
        "Maximum action:",
        action.max().item()
    )

    print(
        "Log probability:"
    )

    print(
        log_probability
    )

    print(
        "Entropy:"
    )

    print(
        entropy
    )

    # ------------------------------------------------------
    # Check that action can be evaluated again
    # ------------------------------------------------------

    new_log_probability, new_entropy = (
        actor.evaluate_action(
            observation,
            action
        )
    )

    print(
        "Evaluated log probability:"
    )

    print(
        new_log_probability
    )

    print(
        "Evaluated entropy:"
    )

    print(
        new_entropy
    )
