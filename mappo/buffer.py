import numpy as np


class MultiAgentRolloutBuffer:

    def __init__(
        self,
        num_agents=2,
        obs_dim=7,
        state_dim=8,
        sequence_length=10
    ):

        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.sequence_length = sequence_length

        # ==================================================
        # RAW TIMESTEP DATA
        # ==================================================

        self.observations = [
            [] for _ in range(num_agents)
        ]

        self.actions = [
            [] for _ in range(num_agents)
        ]

        self.rewards = [
            [] for _ in range(num_agents)
        ]

        self.log_probabilities = [
            [] for _ in range(num_agents)
        ]

        # Episode boundary flags.
        # episode_ends includes both true termination and time-limit truncation.
        # terminateds includes only true MDP terminals.
        self.dones = []
        self.episode_ends = []
        self.terminateds = []

        # V(next_state) used to bootstrap time-limit truncations.
        self.bootstrap_values = []

        # ==================================================
        # CENTRALIZED CRITIC STATE
        # ==================================================
        #
        # The centralized state remains 6-D:
        #
        # [distance_A,
        #  speed_A,
        #  distance_B,
        #  speed_B,
        #  TTC_A,
        #  TTC_B]
        #
        # We are only adding the "other_passed" feature
        # to the decentralized actor observations.
        # ==================================================

        self.states = []

        # One value for each agent
        #
        # values[t] = [V_A, V_B]

        self.values = []

        # ==================================================
        # GAE
        # ==================================================

        self.advantages = [
            [] for _ in range(num_agents)
        ]

        self.returns = [
            [] for _ in range(num_agents)
        ]

    # ======================================================
    # ADD ONE TIMESTEP
    # ======================================================

    def add(
        self,
        observations,
        state,
        actions,
        rewards,
        log_probabilities,
        values,
        done=None,
        terminated=None,
        episode_end=None,
        bootstrap_values=None
    ):

        """
        Store one environment timestep.

        observations:
            [
                observation_A,
                observation_B
            ]

        Each observation now contains 7 features:

            [
                own_distance,
                own_speed,
                other_distance,
                other_speed,
                own_TTC,
                other_TTC,
                other_passed
            ]

        actions:
            [
                action_A,
                action_B
            ]

        rewards:
            [
                reward_A,
                reward_B
            ]

        log_probabilities:
            [
                log_prob_A,
                log_prob_B
            ]

        values:
            [
                V_A,
                V_B
            ]

        state:
            centralized 8-D state.

        done:
            Backward-compatible alias for episode_end.

        terminated:
            True only for a genuine terminal state (collision/success).

        episode_end:
            True whenever the episode stops, including a time-limit truncation.

        bootstrap_values:
            V(next_state) for time-limit truncation; zero for true terminals.
        """

        # --------------------------------------------------
        # Per-agent data
        # --------------------------------------------------

        for agent in range(
            self.num_agents
        ):

            self.observations[agent].append(
                np.asarray(
                    observations[agent],
                    dtype=np.float32
                )
            )

            self.actions[agent].append(
                np.asarray(
                    actions[agent],
                    dtype=np.float32
                )
            )

            self.rewards[agent].append(
                float(rewards[agent])
            )

            self.log_probabilities[agent].append(
                float(
                    log_probabilities[agent]
                )
            )

        # --------------------------------------------------
        # Centralized critic state
        # --------------------------------------------------

        self.states.append(
            np.asarray(
                state,
                dtype=np.float32
            )
        )

        # --------------------------------------------------
        # Centralized critic values
        # --------------------------------------------------

        self.values.append(
            np.asarray(
                values,
                dtype=np.float32
            )
        )

        # --------------------------------------------------
        # Episode termination / truncation
        # --------------------------------------------------

        if episode_end is None:
            episode_end = bool(done) if done is not None else False

        if terminated is None:
            terminated = bool(done) if done is not None else False

        if bootstrap_values is None:
            bootstrap_values = np.zeros(
                self.num_agents,
                dtype=np.float32
            )
        else:
            bootstrap_values = np.asarray(
                bootstrap_values,
                dtype=np.float32
            )

        self.dones.append(bool(episode_end))
        self.episode_ends.append(bool(episode_end))
        self.terminateds.append(bool(terminated))
        self.bootstrap_values.append(bootstrap_values.copy())

    # ======================================================
    # COMPUTE GAE
    # ======================================================

    def compute_returns_and_advantages(
        self,
        last_values,
        gamma=0.99,
        gae_lambda=0.95
    ):

        """
        last_values:

            [V_A(last), V_B(last)]
        """

        values = np.asarray(
            self.values,
            dtype=np.float32
        )

        episode_ends = np.asarray(
            self.episode_ends,
            dtype=np.float32
        )

        terminateds = np.asarray(
            self.terminateds,
            dtype=np.float32
        )

        bootstrap_values = np.asarray(
            self.bootstrap_values,
            dtype=np.float32
        )

        for agent in range(
            self.num_agents
        ):

            rewards = np.asarray(
                self.rewards[agent],
                dtype=np.float32
            )

            advantages = np.zeros_like(
                rewards
            )

            gae = 0.0

            for t in reversed(
                range(len(rewards))
            ):

                # ------------------------------------------
                # Value at next timestep
                # ------------------------------------------

                if episode_ends[t]:
                    # For a true terminal this is zero.
                    # For a time-limit truncation this is V(next_state).
                    next_value = bootstrap_values[t, agent]

                elif t == len(rewards) - 1:
                    # Rollout ended before the environment episode did.
                    next_value = last_values[agent]

                else:
                    next_value = values[t + 1, agent]

                # ------------------------------------------
                # Bootstrap mask
                # ------------------------------------------
                # Time-limit truncation is not an MDP terminal,
                # so it must bootstrap. True termination must not.

                next_non_terminal = 1.0 - terminateds[t]

                # ------------------------------------------
                # TD error
                # ------------------------------------------

                delta = (
                    rewards[t]
                    + gamma
                    * next_value
                    * next_non_terminal
                    - values[t, agent]
                )

                # ------------------------------------------
                # GAE
                # ------------------------------------------
                # Do not let GAE recursion cross an episode boundary.

                gae = (
                    delta
                    + gamma
                    * gae_lambda
                    * (1.0 - episode_ends[t])
                    * gae
                )

                advantages[t] = gae

            # ----------------------------------------------
            # Return
            # ----------------------------------------------

            returns = (
                advantages
                + values[:, agent]
            )

            self.advantages[agent] = (
                advantages
            )

            self.returns[agent] = (
                returns
            )

    # ======================================================
    # FIND EPISODE START
    # ======================================================

    def _find_episode_start(
        self,
        timestep
    ):

        """
        Find the first timestep of the
        current episode.
        """

        for t in range(
            timestep - 1,
            -1,
            -1
        ):

            if self.episode_ends[t]:

                return t + 1

        return 0

    # ======================================================
    # BUILD ONE RECURRENT SEQUENCE
    # ======================================================

    def _build_sequence(
        self,
        data,
        timestep
    ):

        episode_start = (
            self._find_episode_start(
                timestep
            )
        )

        sequence_start = max(
            episode_start,
            timestep
            - self.sequence_length
            + 1
        )

        sequence = data[
            sequence_start:timestep + 1
        ]

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Pad beginning if necessary
        # --------------------------------------------------

        current_length = len(
            sequence
        )

        if current_length < self.sequence_length:

            padding_length = (
                self.sequence_length
                - current_length
            )

            # Repeat first valid observation
            padding = np.repeat(
                sequence[0:1],
                padding_length,
                axis=0
            )

            sequence = np.concatenate(
                [
                    padding,
                    sequence
                ],
                axis=0
            )

        return sequence

    # ======================================================
    # BUILD RECURRENT TRAINING DATA
    # ======================================================

    def get_recurrent_training_data(self):

        num_timesteps = len(
            self.states
        )

        if num_timesteps == 0:

            return None

        result = {

            # Critic
            "states": [],

            # Actor
            "observations": [
                [] for _ in range(
                    self.num_agents
                )
            ],

            # PPO targets
            "values": [],

            "actions": [
                [] for _ in range(
                    self.num_agents
                )
            ],

            "old_log_probabilities": [
                [] for _ in range(
                    self.num_agents
                )
            ],

            "advantages": [
                [] for _ in range(
                    self.num_agents
                )
            ],

            "returns": [
                [] for _ in range(
                    self.num_agents
                )
            ],

            "timesteps": []
        }

        # ==================================================
        # ONE TRAINING SAMPLE PER TIMESTEP
        # ==================================================

        for timestep in range(
            num_timesteps
        ):

            # ----------------------------------------------
            # Centralized critic sequence
            # ----------------------------------------------

            state_sequence = (
                self._build_sequence(
                    self.states,
                    timestep
                )
            )

            result["states"].append(
                state_sequence
            )

            # ----------------------------------------------
            # Critic value at current timestep
            # ----------------------------------------------

            result["values"].append(
                np.asarray(
                    self.values[timestep],
                    dtype=np.float32
                )
            )

            # ----------------------------------------------
            # Each agent
            # ----------------------------------------------

            for agent in range(
                self.num_agents
            ):

                observation_sequence = (
                    self._build_sequence(
                        self.observations[agent],
                        timestep
                    )
                )

                result[
                    "observations"
                ][agent].append(
                    observation_sequence
                )

                # ------------------------------------------
                # Target action at current timestep
                # ------------------------------------------

                result[
                    "actions"
                ][agent].append(
                    self.actions[agent][timestep]
                )

                # ------------------------------------------
                # Old policy probability
                # ------------------------------------------

                result[
                    "old_log_probabilities"
                ][agent].append(
                    self.log_probabilities[
                        agent
                    ][timestep]
                )

                # ------------------------------------------
                # Advantage
                # ------------------------------------------

                result[
                    "advantages"
                ][agent].append(
                    self.advantages[
                        agent
                    ][timestep]
                )

                # ------------------------------------------
                # Return
                # ------------------------------------------

                result[
                    "returns"
                ][agent].append(
                    self.returns[
                        agent
                    ][timestep]
                )

            result["timesteps"].append(
                timestep
            )

        # ==================================================
        # CONVERT TO NUMPY
        # ==================================================

        result["states"] = np.asarray(
            result["states"],
            dtype=np.float32
        )

        for agent in range(
            self.num_agents
        ):

            result[
                "observations"
            ][agent] = np.asarray(
                result[
                    "observations"
                ][agent],
                dtype=np.float32
            )

            result[
                "actions"
            ][agent] = np.asarray(
                result[
                    "actions"
                ][agent],
                dtype=np.float32
            )

            result[
                "old_log_probabilities"
            ][agent] = np.asarray(
                result[
                    "old_log_probabilities"
                ][agent],
                dtype=np.float32
            )

            result[
                "advantages"
            ][agent] = np.asarray(
                result[
                    "advantages"
                ][agent],
                dtype=np.float32
            )

            result[
                "returns"
            ][agent] = np.asarray(
                result[
                    "returns"
                ][agent],
                dtype=np.float32
            )

        result["timesteps"] = np.asarray(
            result["timesteps"],
            dtype=np.int32
        )

        return result

    # ======================================================
    # RAW DATA
    # ======================================================

    def get(self):

        data = {

            "states": np.asarray(
                self.states,
                dtype=np.float32
            ),

            "values": np.asarray(
                self.values,
                dtype=np.float32
            ),

            # `dones` is kept as an episode-boundary alias.
            "dones": np.asarray(
                self.dones,
                dtype=np.float32
            ),

            "episode_ends": np.asarray(
                self.episode_ends,
                dtype=np.float32
            ),

            "terminateds": np.asarray(
                self.terminateds,
                dtype=np.float32
            ),

            "bootstrap_values": np.asarray(
                self.bootstrap_values,
                dtype=np.float32
            )
        }

        for agent in range(
            self.num_agents
        ):

            data[
                f"observations_{agent}"
            ] = np.asarray(
                self.observations[agent],
                dtype=np.float32
            )

            data[
                f"actions_{agent}"
            ] = np.asarray(
                self.actions[agent],
                dtype=np.float32
            )

            data[
                f"rewards_{agent}"
            ] = np.asarray(
                self.rewards[agent],
                dtype=np.float32
            )

            data[
                f"log_probabilities_{agent}"
            ] = np.asarray(
                self.log_probabilities[agent],
                dtype=np.float32
            )

            data[
                f"advantages_{agent}"
            ] = np.asarray(
                self.advantages[agent],
                dtype=np.float32
            )

            data[
                f"returns_{agent}"
            ] = np.asarray(
                self.returns[agent],
                dtype=np.float32
            )

        return data

    # ======================================================
    # CLEAR
    # ======================================================

    def clear(self):

        self.observations = [
            [] for _ in range(
                self.num_agents
            )
        ]

        self.actions = [
            [] for _ in range(
                self.num_agents
            )
        ]

        self.rewards = [
            [] for _ in range(
                self.num_agents
            )
        ]

        self.log_probabilities = [
            [] for _ in range(
                self.num_agents
            )
        ]

        self.dones = []
        self.episode_ends = []
        self.terminateds = []
        self.bootstrap_values = []

        self.states = []
        self.values = []

        self.advantages = [
            [] for _ in range(
                self.num_agents
            )
        ]

        self.returns = [
            [] for _ in range(
                self.num_agents
            )
        ]

    def __len__(self):

        return len(
            self.states
        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print(
        "======================================"
    )

    print(
        "TWO-VALUE EPISODE-AWARE BUFFER TEST"
    )

    print(
        "======================================"
    )

    buffer = MultiAgentRolloutBuffer(
        num_agents=2,
        obs_dim=7,
        state_dim=8,
        sequence_length=10
    )

    # ======================================================
    # EPISODE 1
    # ======================================================

    for t in range(12):

        # --------------------------------------------------
        # 7-dimensional observations
        # --------------------------------------------------

        obs_A = np.ones(7) * t

        obs_B = np.ones(7) * (
            100 + t
        )

        # Centralized critic is 8-D
        state = np.ones(8) * (
            200 + t
        )

        buffer.add(

            observations=[
                obs_A,
                obs_B
            ],

            state=state,

            actions=[
                np.array([0.5]),
                np.array([-0.3])
            ],

            rewards=[
                1.0,
                0.8
            ],

            log_probabilities=[
                -0.5,
                -0.6
            ],

            values=[
                0.5,
                0.7
            ],

            done=(t == 11)
        )

    # ======================================================
    # EPISODE 2
    # ======================================================

    for t in range(8):

        # 7-dimensional observations
        obs_A = np.ones(7) * (
            1000 + t
        )

        obs_B = np.ones(7) * (
            2000 + t
        )

        # Centralized critic is 8-D
        state = np.ones(8) * (
            3000 + t
        )

        buffer.add(

            observations=[
                obs_A,
                obs_B
            ],

            state=state,

            actions=[
                np.array([0.7]),
                np.array([-0.2])
            ],

            rewards=[
                1.2,
                0.9
            ],

            log_probabilities=[
                -0.4,
                -0.5
            ],

            values=[
                0.6,
                0.8
            ],

            done=(t == 7)
        )

    # ======================================================
    # BASIC SHAPES
    # ======================================================

    print()
    print(
        "Total timesteps:",
        len(buffer)
    )

    raw_data = buffer.get()

    print()
    print(
        "Raw observations A:",
        raw_data["observations_0"].shape
    )

    print(
        "Raw observations B:",
        raw_data["observations_1"].shape
    )

    print(
        "Raw centralized states:",
        raw_data["states"].shape
    )

    print(
        "Critic values:",
        raw_data["values"].shape
    )

    # ======================================================
    # GAE
    # ======================================================

    buffer.compute_returns_and_advantages(

        last_values=[
            0.0,
            0.0
        ],

        gamma=0.99,

        gae_lambda=0.95
    )

    # ======================================================
    # RECURRENT TRAINING DATA
    # ======================================================

    training_data = (
        buffer.get_recurrent_training_data()
    )

    print()
    print(
        "Recurrent critic sequences:",
        training_data["states"].shape
    )

    print(
        "Car A recurrent sequences:",
        training_data[
            "observations"
        ][0].shape
    )

    print(
        "Car B recurrent sequences:",
        training_data[
            "observations"
        ][1].shape
    )

    print(
        "Car A actions:",
        training_data[
            "actions"
        ][0].shape
    )

    print(
        "Car B actions:",
        training_data[
            "actions"
        ][1].shape
    )

    # ======================================================
    # EPISODE BOUNDARY CHECK
    # ======================================================

    print()
    print(
        "Checking episode boundary..."
    )

    sequence = training_data[
        "observations"
    ][0][12]

    print(
        "Car A sequence at beginning "
        "of Episode 2:"
    )

    print(
        sequence[:, 0]
    )

    # ======================================================
    # VALUE CHECK
    # ======================================================

    print()
    print(
        "Example critic values:"
    )

    print(
        raw_data["values"][:3]
    )

    print()
    print(
        "======================================"
    )

    print(
        "TEST COMPLETE"
    )

    print(
        "======================================"
    )

