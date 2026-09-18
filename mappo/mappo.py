import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from actor import LSTMActor
from critic import LSTMCritic


class MAPPO:

    def __init__(
        self,
        obs_dim=7,
        state_dim=8,
        action_dim=1,
        num_agents=2,
        sequence_length=10,
        action_limit=3.0,
        actor_lr=3e-4,
        critic_lr=3e-4,
        clip_epsilon=0.2,
        entropy_coefficient=0.01
    ):

        self.num_agents = num_agents
        self.sequence_length = sequence_length

        self.clip_epsilon = clip_epsilon
        self.entropy_coefficient = entropy_coefficient

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {self.device}"
        )

        # ----------------------------------------------------
        # SHARED ACTOR
        # ----------------------------------------------------

        self.actor = LSTMActor(

            obs_dim=obs_dim,

            hidden_dim=128,

            fc_dim=64,

            action_dim=action_dim,

            action_limit=action_limit

        ).to(self.device)

        # ----------------------------------------------------
        # CENTRALIZED CRITIC
        # ----------------------------------------------------

        self.critic = LSTMCritic(

            state_dim=state_dim,

            hidden_dim=128,

            fc_dim=64,

            num_agents=num_agents

        ).to(self.device)

        # ----------------------------------------------------
        # OPTIMIZERS
        # ----------------------------------------------------

        self.actor_optimizer = optim.Adam(

            self.actor.parameters(),

            lr=actor_lr

        )

        self.critic_optimizer = optim.Adam(

            self.critic.parameters(),

            lr=critic_lr

        )

    # ========================================================
    # SELECT ACTIONS
    # ========================================================

    def select_actions(
        self,
        observations
    ):

        """
        Select actions for all agents.

        observations:

            {
                0: [sequence_length, obs_dim],
                1: [sequence_length, obs_dim]
            }

        With the current environment:

            obs_dim = 7

        Each observation contains:

            [
                own_distance,
                own_speed,
                other_distance,
                other_speed,
                own_TTC,
                other_TTC,
                other_passed
            ]

        Returns:

            actions
            log_probabilities
            entropies
        """

        actions = {}

        log_probabilities = {}

        entropies = {}

        # ----------------------------------------------------
        # PROCESS EACH AGENT
        # ----------------------------------------------------

        for agent_id in range(
            self.num_agents
        ):

            observation = observations[
                agent_id
            ]

            # Convert numpy array → tensor
            observation = torch.tensor(

                observation,

                dtype=torch.float32,

                device=self.device

            )

            # Add batch dimension
            #
            # [10, 7]
            #
            # becomes
            #
            # [1, 10, 7]

            if observation.dim() == 2:

                observation = observation.unsqueeze(0)

            # ------------------------------------------------
            # ACTOR
            # ------------------------------------------------

            with torch.no_grad():

                (
                    action,
                    log_probability,
                    entropy

                ) = self.actor.get_action(
                    observation
                )

            # ------------------------------------------------
            # REMOVE BATCH DIMENSION
            # ------------------------------------------------

            action = action.squeeze(0)

            log_probability = (
                log_probability.squeeze(0)
            )

            entropy = entropy.squeeze(0)

            # ------------------------------------------------
            # MOVE TO CPU
            # ------------------------------------------------

            actions[agent_id] = (
                action.cpu().numpy()
            )

            log_probabilities[agent_id] = (
                log_probability.cpu().item()
            )

            entropies[agent_id] = (
                entropy.cpu().item()
            )

        return (
            actions,
            log_probabilities,
            entropies
        )

    # ========================================================
    # GET CENTRALIZED VALUE
    # ========================================================

    def get_value(
        self,
        state_sequence
    ):

        """
        Obtain centralized critic values.

        Input:

            [sequence_length, state_dim]

        Current state_dim:

            8

        State:

            [
        distance_A,
        speed_A,
        distance_B,
        speed_B,
        TTC_A,
        TTC_B,
        A_passed,
        B_passed
        ]

        Output:

            [V_A, V_B]
        """

        state_sequence = torch.tensor(

            state_sequence,

            dtype=torch.float32,

            device=self.device

        )

        # Add batch dimension

        if state_sequence.dim() == 2:

            state_sequence = (
                state_sequence.unsqueeze(0)
            )

        # ----------------------------------------------------
        # CRITIC
        # ----------------------------------------------------

        with torch.no_grad():

            values = self.critic(
                state_sequence
            )

        # values shape:
        #
        # [1, 2]
        #
        # Convert to:
        #
        # [2]

        values = values.squeeze(0)

        return values.cpu().numpy()

    # ========================================================
    # UPDATE MAPPO
    # ========================================================

    def update_from_buffer(
        self,
        buffer,
        epochs=10
    ):

        """
        Perform MAPPO/PPO update using the collected
        rollout data.
        """

        # ----------------------------------------------------
        # GET RECURRENT TRAINING DATA
        # ----------------------------------------------------

        data = (
            buffer.get_recurrent_training_data()
        )

        # ----------------------------------------------------
        # OBSERVATIONS
        # ----------------------------------------------------

        observations_A = torch.tensor(

            data["observations"][0],

            dtype=torch.float32,

            device=self.device

        )

        observations_B = torch.tensor(

            data["observations"][1],

            dtype=torch.float32,

            device=self.device

        )

        # ----------------------------------------------------
        # CENTRALIZED STATES
        # ----------------------------------------------------

        states = torch.tensor(

            data["states"],

            dtype=torch.float32,

            device=self.device

        )

        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        actions_A = torch.tensor(

            data["actions"][0],

            dtype=torch.float32,

            device=self.device

        )

        actions_B = torch.tensor(

            data["actions"][1],

            dtype=torch.float32,

            device=self.device

        )

        # ----------------------------------------------------
        # OLD LOG PROBABILITIES
        # ----------------------------------------------------

        old_log_probs_A = torch.tensor(

            data["old_log_probabilities"][0],

            dtype=torch.float32,

            device=self.device

        )

        old_log_probs_B = torch.tensor(

            data["old_log_probabilities"][1],

            dtype=torch.float32,

            device=self.device

        )

        # ----------------------------------------------------
        # ADVANTAGES
        # ----------------------------------------------------

        advantages_A = torch.tensor(

            data["advantages"][0],

            dtype=torch.float32,

            device=self.device

        )

        advantages_B = torch.tensor(

            data["advantages"][1],

            dtype=torch.float32,

            device=self.device

        )

        # ----------------------------------------------------
        # RETURNS
        # ----------------------------------------------------

        returns_A = torch.tensor(

            data["returns"][0],

            dtype=torch.float32,

            device=self.device

        )

        returns_B = torch.tensor(

            data["returns"][1],

            dtype=torch.float32,

            device=self.device

        )

        # ====================================================
        # TRAINING DIAGNOSTICS
        # ====================================================

        raw_advantage_mean = (
            (advantages_A.mean() + advantages_B.mean()) / 2.0
        ).item()

        raw_advantage_std = (
            (advantages_A.std() + advantages_B.std()) / 2.0
        ).item()

        raw_value_mean = float(np.mean(data["values"]))

        raw_return_mean = (
            (returns_A.mean() + returns_B.mean()) / 2.0
        ).item()


        # ====================================================
        # NORMALIZE ADVANTAGES
        # ====================================================

        advantages_A = (

            advantages_A
            - advantages_A.mean()

        ) / (

            advantages_A.std() + 1e-8

        )

        advantages_B = (

            advantages_B
            - advantages_B.mean()

        ) / (

            advantages_B.std() + 1e-8

        )

        # ====================================================
        # TRAINING
        # ====================================================

        actor_loss_value = 0.0
        critic_loss_value = 0.0
        approx_kl_value = 0.0
        entropy_value = 0.0

        for epoch in range(
            epochs
        ):

            # =================================================
            # ACTOR — CAR A
            # =================================================

            new_log_probs_A, entropy_A = (
                self.actor.evaluate_action(

                    observations_A,

                    actions_A

                )
            )

            # =================================================
            # ACTOR — CAR B
            # =================================================

            new_log_probs_B, entropy_B = (
                self.actor.evaluate_action(

                    observations_B,

                    actions_B

                )
            )

            # =================================================
            # PPO RATIO — CAR A
            # =================================================

            ratio_A = torch.exp(

                new_log_probs_A
                - old_log_probs_A

            )

            # =================================================
            # PPO RATIO — CAR B
            # =================================================

            ratio_B = torch.exp(

                new_log_probs_B
                - old_log_probs_B

            )

            # =================================================
            # PPO DIAGNOSTICS
            # =================================================

            approx_kl = (
                (
                    old_log_probs_A - new_log_probs_A
                ).mean()
                +
                (
                    old_log_probs_B - new_log_probs_B
                ).mean()
            ) / 2.0

            mean_entropy = (
                entropy_A.mean() + entropy_B.mean()
            ) / 2.0


            # =================================================
            # CLIPPED OBJECTIVE — CAR A
            # =================================================

            unclipped_A = (

                ratio_A
                * advantages_A

            )

            clipped_A = (

                torch.clamp(

                    ratio_A,

                    1.0 - self.clip_epsilon,

                    1.0 + self.clip_epsilon

                )
                * advantages_A

            )

            policy_loss_A = -torch.min(

                unclipped_A,

                clipped_A

            ).mean()

            # =================================================
            # CLIPPED OBJECTIVE — CAR B
            # =================================================

            unclipped_B = (

                ratio_B
                * advantages_B

            )

            clipped_B = (

                torch.clamp(

                    ratio_B,

                    1.0 - self.clip_epsilon,

                    1.0 + self.clip_epsilon

                )
                * advantages_B

            )

            policy_loss_B = -torch.min(

                unclipped_B,

                clipped_B

            ).mean()

            # =================================================
            # ENTROPY
            # =================================================

            entropy_loss = -self.entropy_coefficient * (

                entropy_A.mean()
                + entropy_B.mean()

            ) / 2.0

            # =================================================
            # TOTAL ACTOR LOSS
            # =================================================

            actor_loss = (

                policy_loss_A
                + policy_loss_B

            ) / 2.0

            actor_loss = (
                actor_loss
                + entropy_loss
            )

            # =================================================
            # UPDATE ACTOR
            # =================================================

            self.actor_optimizer.zero_grad()

            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(

                self.actor.parameters(),

                max_norm=0.5

            )

            self.actor_optimizer.step()

            # =================================================
            # CENTRALIZED CRITIC
            # =================================================

            predicted_values = self.critic(
                states
            )

            # predicted_values:
            #
            # [batch, 2]
            #
            # column 0 → V_A
            # column 1 → V_B

            predicted_A = (
                predicted_values[:, 0]
            )

            predicted_B = (
                predicted_values[:, 1]
            )

            # =================================================
            # CRITIC LOSS
            # =================================================

            critic_loss_A = torch.mean(

                (
                    returns_A
                    - predicted_A
                ) ** 2

            )

            critic_loss_B = torch.mean(

                (
                    returns_B
                    - predicted_B
                ) ** 2

            )

            critic_loss = (

                critic_loss_A
                + critic_loss_B

            ) / 2.0

            # =================================================
            # UPDATE CRITIC
            # =================================================

            self.critic_optimizer.zero_grad()

            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(

                self.critic.parameters(),

                max_norm=0.5

            )

            self.critic_optimizer.step()

            # =================================================
            # SAVE LOSS / DIAGNOSTICS
            # =================================================

            actor_loss_value = (
                actor_loss.item()
            )

            critic_loss_value = (
                critic_loss.item()
            )

            approx_kl_value = (
                approx_kl.detach().item()
            )

            entropy_value = (
                mean_entropy.detach().item()
            )

        # ====================================================
        # RETURN TRAINING INFORMATION
        # ====================================================

        return {

            "actor_loss":
                actor_loss_value,

            "critic_loss":
                critic_loss_value,

            "approx_kl":
                approx_kl_value,

            "entropy":
                entropy_value,

            "adv_mean":
                raw_advantage_mean,

            "adv_std":
                raw_advantage_std,

            "value_mean":
                raw_value_mean,

            "return_mean":
                raw_return_mean

        }

