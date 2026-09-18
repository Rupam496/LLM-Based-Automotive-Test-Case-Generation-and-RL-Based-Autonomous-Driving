import sys
import os
import csv
from collections import deque

import numpy as np
import torch


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT PROJECT COMPONENTS
# ============================================================

from carla_env import CarlaIntersectionEnv
from buffer import MultiAgentRolloutBuffer
from mappo import MAPPO


# ============================================================
# CONFIGURATION
# ============================================================

# For the first sanity test, keep this at 10.
#
# After the sanity test succeeds, change it to:
#
# NUM_EPISODES = 500

NUM_EPISODES = 1500


# Number of complete episodes collected before
# performing one PPO/MAPPO update.

ROLLOUT_EPISODES = 10


SEQUENCE_LENGTH = 10

MAPPO_EPOCHS = 5

GAMMA = 0.99

GAE_LAMBDA = 0.95

ACTION_LIMIT = 3.0

CSV_FILE = "training_metrics.csv"

MODEL_FILE = "mappo_final.pth"

ROLLING_WINDOW = 10

DIAGNOSTIC_EPISODES = {
    1, 100, 200, 300, 400, 500,
    550, 600, 700, 800, 900, 1000
}

DIAGNOSTIC_CSV_FILE = "trajectory_diagnostics.csv"


# ============================================================
# DIMENSIONS
# ============================================================

# Actor observation:
#
# Both agents use the same SELF / OTHER semantic ordering:
#
# [
#     own_distance,
#     own_speed,
#     own_TTC,
#     other_distance,
#     other_speed,
#     other_TTC,
#     other_passed
# ]
#
# Therefore:
#
# Agent A:
# [
#     dA,
#     vA,
#     TTC_A,
#     dB,
#     vB,
#     TTC_B,
#     B_passed
# ]
#
# Agent B:
# [
#     dB,
#     vB,
#     TTC_B,
#     dA,
#     vA,
#     TTC_A,
#     A_passed
# ]

OBS_DIM = 7


# Centralized critic state:
#
# [
#     dA,
#     vA,
#     dB,
#     vB,
#     TTC_A,
#     TTC_B,
#     A_passed,
#     B_passed
# ]

STATE_DIM = 8


NUM_AGENTS = 2


# ============================================================
# OBSERVATION NORMALIZATION
# ============================================================

def normalize_observation(obs):

    """
    Normalize the 7-dimensional actor observation.

    Input:

        [
            own_distance,
            own_speed,
            own_TTC,
            other_distance,
            other_speed,
            other_TTC,
            other_passed
        ]

    Normalization:

        distance      -> / 50
        speed         -> / 12
        TTC           -> / 100
        passed        -> 0 or 1

    The values are clipped so that unexpected values
    cannot produce very large neural-network inputs.
    """

    obs = np.asarray(
        obs,
        dtype=np.float32
    ).copy()

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    obs[0] = np.clip(
        obs[0] / 50.0,
        -1.0,
        1.0
    )

    obs[3] = np.clip(
        obs[3] / 50.0,
        -1.0,
        1.0
    )

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    obs[1] = np.clip(
        obs[1] / 12.0,
        0.0,
        1.0
    )

    obs[4] = np.clip(
        obs[4] / 12.0,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # TTC
    # --------------------------------------------------------

    obs[2] = np.clip(
        obs[2] / 100.0,
        0.0,
        1.0
    )

    obs[5] = np.clip(
        obs[5] / 100.0,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Other vehicle passed flag
    # --------------------------------------------------------

    obs[6] = np.clip(
        obs[6],
        0.0,
        1.0
    )

    return obs.astype(
        np.float32
    )


# ============================================================
# LSTM HISTORY
# ============================================================

def create_initial_history(
    observation,
    sequence_length
):

    """
    Create an initial history for the LSTM.

    The initial observation is repeated so that the LSTM
    always receives a sequence of length SEQUENCE_LENGTH.
    """

    history = deque(
        maxlen=sequence_length
    )

    observation = np.asarray(
        observation,
        dtype=np.float32
    )

    for _ in range(sequence_length):

        history.append(
            observation.copy()
        )

    return history


def history_to_array(history):

    """
    Convert a deque history into a numpy array.

    Output shape:

        [sequence_length, feature_dimension]
    """

    return np.asarray(
        history,
        dtype=np.float32
    )


# ============================================================
# CENTRALIZED CRITIC STATE
# ============================================================

def build_centralized_state(
    obs_A,
    obs_B
):

    """
    Build the centralized 8-dimensional critic state.

    Agent A observation:

        [
            dA,
            vA,
            TTC_A,
            dB,
            vB,
            TTC_B,
            B_passed
        ]

    Agent B observation:

        [
            dB,
            vB,
            TTC_B,
            dA,
            vA,
            TTC_A,
            A_passed
        ]

    Centralized critic state:

        [
            dA,
            vA,
            dB,
            vB,
            TTC_A,
            TTC_B,
            A_passed,
            B_passed
        ]

    The first six values are already normalized because
    normalize_observation() is called before this function.
    """

    state = np.array(

        [

            # ------------------------------------------------
            # Car A
            # ------------------------------------------------

            obs_A[0],
            obs_A[1],

            # ------------------------------------------------
            # Car B
            # ------------------------------------------------

            obs_B[0],
            obs_B[1],

            # ------------------------------------------------
            # TTC
            # ------------------------------------------------

            obs_A[2],
            obs_A[5],

            # ------------------------------------------------
            # Passed flags
            # ------------------------------------------------

            # A passed
            obs_B[6],

            # B passed
            obs_A[6]

        ],

        dtype=np.float32

    )

    return state


# ============================================================
# CREATE ROLLOUT BUFFER
# ============================================================

def create_rollout_buffer():

    """
    Create one buffer for multiple complete episodes.

    Important:

        The buffer is NOT recreated after every episode.

        It is recreated only after a MAPPO update.

    Therefore, several episodes are stored together.
    """

    return MultiAgentRolloutBuffer(

        num_agents=NUM_AGENTS,

        obs_dim=OBS_DIM,

        state_dim=STATE_DIM,

        sequence_length=SEQUENCE_LENGTH

    )


# ============================================================
# TRAINING
# ============================================================

def train():

    print()
    print("======================================")
    print("CARLA + MAPPO TRAINING")
    print("======================================")

    print()

    print(
        f"Total episodes:       "
        f"{NUM_EPISODES}"
    )

    print(
        f"Episodes per update:  "
        f"{ROLLOUT_EPISODES}"
    )

    print(
        f"Actor observation:    "
        f"{OBS_DIM}D"
    )

    print(
        f"Critic state:         "
        f"{STATE_DIM}D"
    )

    print(
        f"Sequence length:      "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"MAPPO epochs:         "
        f"{MAPPO_EPOCHS}"
    )

    print()


    # ========================================================
    # CREATE CARLA ENVIRONMENT
    # ========================================================

    env = CarlaIntersectionEnv()


    # ========================================================
    # CREATE MAPPO
    # ========================================================

    mappo = MAPPO(

        obs_dim=OBS_DIM,

        state_dim=STATE_DIM,

        action_dim=1,

        num_agents=NUM_AGENTS,

        sequence_length=SEQUENCE_LENGTH,

        action_limit=ACTION_LIMIT

    )


    # ========================================================
    # CREATE CSV FILE
    # ========================================================

    with open(

        CSV_FILE,

        "w",

        newline=""

    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "episode",

            "reward_A",

            "reward_B",

            "episode_length",

            "collision",

            "A_passed",

            "B_passed",

            "success",

            "time_limit",

            "average_ttc",

            "actor_loss",

            "critic_loss",

            "approx_kl",
            "entropy",
            "adv_mean",
            "adv_std",
            "value_mean",
            "return_mean",
            "A_recovery_time_steps",
            "B_recovery_time_steps",
            "deadlock"

        ])


    # ========================================================
    # CREATE TRAJECTORY DIAGNOSTIC CSV
    # ========================================================

    with open(
        DIAGNOSTIC_CSV_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "episode", "step",
            "action_A", "action_B",
            "distance_A", "speed_A", "ttc_A",
            "distance_B", "speed_B", "ttc_B",
            "A_passed", "B_passed",
            "collision",
            "A_stopped", "B_stopped",
            "A_other_passed", "B_other_passed",
            "A_pass_step", "B_pass_step",
            "A_recovery_time_steps", "B_recovery_time_steps"
        ])


    # ========================================================
    # TRAINING STATISTICS
    # ========================================================

    episode_rewards_A = []

    episode_rewards_B = []

    episode_lengths = []

    episode_collisions = []

    episode_successes = []

    episode_time_limits = []

    episode_avg_ttc = []


    # ========================================================
    # CREATE ONE ROLLOUT BUFFER
    # ========================================================

    buffer = create_rollout_buffer()

    episodes_in_rollout = 0


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    try:

        for episode in range(

            NUM_EPISODES

        ):

            print()
            print(
                "======================================"
            )

            print(

                f"Episode "
                f"{episode + 1}/"
                f"{NUM_EPISODES}"

            )

            print(

                f"Rollout progress: "
                f"{episodes_in_rollout + 1}/"
                f"{ROLLOUT_EPISODES}"

            )

            print(
                "======================================"
            )


            # =================================================
            # RESET ENVIRONMENT
            # =================================================

            obs_A, obs_B = env.reset()


            # =================================================
            # NORMALIZE INITIAL OBSERVATIONS
            # =================================================

            obs_A = normalize_observation(
                obs_A
            )

            obs_B = normalize_observation(
                obs_B
            )


            # =================================================
            # INITIAL ACTOR HISTORY
            # =================================================

            history_A = create_initial_history(

                obs_A,

                SEQUENCE_LENGTH

            )

            history_B = create_initial_history(

                obs_B,

                SEQUENCE_LENGTH

            )


            # =================================================
            # INITIAL CENTRALIZED STATE
            # =================================================

            state = build_centralized_state(

                obs_A,

                obs_B

            )


            # =================================================
            # INITIAL CRITIC HISTORY
            # =================================================

            state_history = create_initial_history(

                state,

                SEQUENCE_LENGTH

            )


            # =================================================
            # EPISODE VARIABLES
            # =================================================

            total_reward_A = 0.0

            total_reward_B = 0.0

            ttc_sum = 0.0

            ttc_count = 0

            done = False

            diagnostic_episode = (
                episode + 1 in DIAGNOSTIC_EPISODES
            )

            A_stopped_steps = 0
            B_stopped_steps = 0


            # =================================================
            # PASSED-STATUS DIAGNOSTIC FLAGS
            # =================================================

            A_passed_reported = False

            B_passed_reported = False

            # Recovery diagnostics
            A_pass_step = None
            B_pass_step = None
            A_recovery_time_steps = None
            B_recovery_time_steps = None


            # =================================================
            # EPISODE LOOP
            # =================================================

            while not done:


                # =============================================
                # ACTOR HISTORY
                # =============================================

                sequence_A = history_to_array(
                    history_A
                )

                sequence_B = history_to_array(
                    history_B
                )


                # =============================================
                # CRITIC HISTORY
                # =============================================

                state_sequence = history_to_array(
                    state_history
                )


                # =============================================
                # GET CENTRALIZED CRITIC VALUE
                # =============================================

                values = mappo.get_value(

                    state_sequence

                )


                # =============================================
                # GET ACTIONS
                # =============================================

                (
                    actions,
                    log_probabilities,
                    entropies

                ) = mappo.select_actions(

                    {

                        0: sequence_A,

                        1: sequence_B

                    }

                )


                # =============================================
                # EXTRACT ACTION A
                # =============================================

                action_A = float(

                    np.asarray(
                        actions[0]
                    ).reshape(-1)[0]

                )


                # =============================================
                # EXTRACT ACTION B
                # =============================================

                action_B = float(

                    np.asarray(
                        actions[1]
                    ).reshape(-1)[0]

                )


                # =============================================
                # SAFETY CLAMP ACTIONS
                # =============================================

                action_A = np.clip(

                    action_A,

                    -ACTION_LIMIT,

                    ACTION_LIMIT

                )

                action_B = np.clip(

                    action_B,

                    -ACTION_LIMIT,

                    ACTION_LIMIT

                )


                # =============================================
                # SNAPSHOT CURRENT TRANSITION STATE
                # =============================================
                #
                # PPO/MAPPO must store:
                #
                #   (obs_t, state_t, action_t, reward_t,
                #    old_log_prob_t, V(s_t))
                #
                # env.step() produces the NEXT observation/state.
                # Therefore preserve the current observation and
                # centralized state before advancing CARLA.
                #
                # This is critical for PPO ratio consistency:
                #
                #   old_log_prob = log pi_old(a_t | obs_t)
                #
                # and the updated policy must evaluate the same
                # action at the same observation obs_t.
                # =============================================

                current_obs_A = np.asarray(
                    obs_A,
                    dtype=np.float32
                ).copy()

                current_obs_B = np.asarray(
                    obs_B,
                    dtype=np.float32
                ).copy()

                current_state = np.asarray(
                    state,
                    dtype=np.float32
                ).copy()

                # Snapshot pass status BEFORE env.step().
                # env.step() may update these flags during this transition.
                was_passed_A = bool(env.passed_A)
                was_passed_B = bool(env.passed_B)

                # =============================================
                # STEP CARLA
                # =============================================

                (
                    next_obs_A,
                    next_obs_B,
                    reward_A,
                    reward_B,
                    done,
                    info

                ) = env.step(

                    action_A,

                    action_B

                )

                # =============================================
                # TERMINATION VS TIME-LIMIT TRUNCATION
                # =============================================

                terminated = bool(
                    info.get("terminated", False)
                )

                truncated = bool(
                    info.get("truncated", False)
                )

                episode_end = (
                    terminated or truncated
                )

                # Keep `done` for controlling the episode loop.
                done = episode_end

                # =============================================
                # RECOVERY / PASS EVENT TRACKING
                # =============================================
                #
                # IMPORTANT:
                # env.step() has already detected newly_passed_A
                # and newly_passed_B and placed those flags in info.
                #
                # Record the first crossing step BEFORE writing the
                # diagnostic CSV. The previous version wrote the CSV
                # first, so the row on which the pass happened still
                # contained None/NaN recovery values.
                # =============================================

                newly_passed_A = (
                    (not was_passed_A and bool(env.passed_A))
                    or bool(info.get("newly_passed_A", False))
                )

                newly_passed_B = (
                    (not was_passed_B and bool(env.passed_B))
                    or bool(info.get("newly_passed_B", False))
                )

                if newly_passed_A and A_pass_step is None:
                    A_pass_step = int(env.step_count)

                if newly_passed_B and B_pass_step is None:
                    B_pass_step = int(env.step_count)

                # Recovery time is the delay from the first vehicle's
                # crossing to the second vehicle's crossing. The first
                # vehicle therefore has recovery time 0, while the
                # second vehicle gets the positive delay.
                if (
                    A_pass_step is not None
                    and B_pass_step is not None
                ):
                    gap = abs(A_pass_step - B_pass_step)
                    A_recovery_time_steps = gap
                    B_recovery_time_steps = gap

                # =============================================
                # SAVE RAW TRAJECTORY DIAGNOSTICS
                # =============================================

                if diagnostic_episode:

                    raw_A = np.asarray(
                        next_obs_A, dtype=np.float32
                    )
                    raw_B = np.asarray(
                        next_obs_B, dtype=np.float32
                    )

                    A_stopped = bool(
                        raw_A[0] > 2.0 and raw_A[1] < 0.5
                    )
                    B_stopped = bool(
                        raw_B[0] > 2.0 and raw_B[1] < 0.5
                    )

                    A_stopped_steps += int(A_stopped)
                    B_stopped_steps += int(B_stopped)

                    with open(
                        DIAGNOSTIC_CSV_FILE,
                        "a",
                        newline=""
                    ) as file:

                        writer = csv.writer(file)
                        writer.writerow([
                            episode + 1,
                            env.step_count,
                            action_A,
                            action_B,
                            raw_A[0],
                            raw_A[1],
                            raw_A[2],
                            raw_B[0],
                            raw_B[1],
                            raw_B[2],
                            bool(env.passed_A),
                            bool(env.passed_B),
                            bool(
                                env.collision_A or env.collision_B
                            ),
                            A_stopped,
                            B_stopped,
                            bool(env.passed_B),
                            bool(env.passed_A),
                            A_pass_step,
                            B_pass_step,
                            A_recovery_time_steps,
                            B_recovery_time_steps
                        ])


                # =============================================
                # NORMALIZE NEXT OBSERVATIONS
                # =============================================

                next_obs_A = normalize_observation(
                    next_obs_A
                )

                next_obs_B = normalize_observation(
                    next_obs_B
                )


                # =============================================
                # ACCUMULATE REWARDS
                # =============================================

                total_reward_A += reward_A

                total_reward_B += reward_B


                # =============================================
                # TTC STATISTICS
                # =============================================

                ttc_A = info["ttc_A"]

                ttc_B = info["ttc_B"]


                if np.isfinite(ttc_A):

                    ttc_sum += ttc_A

                    ttc_count += 1


                if np.isfinite(ttc_B):

                    ttc_sum += ttc_B

                    ttc_count += 1


                # =============================================
                # UPDATE OBSERVATIONS
                # =============================================

                obs_A = next_obs_A

                obs_B = next_obs_B


                # =============================================
                # CHECK PASSED-VEHICLE INFORMATION
                # =============================================

                if (

                    env.passed_A

                    and

                    not A_passed_reported

                ):

                    print()

                    print(
                        "========== A PASSED =========="
                    )

                    print(
                        "B observation after A passed:",
                        obs_B
                    )

                    print(
                        "B other_passed feature:",
                        obs_B[6]
                    )

                    A_passed_reported = True


                if (

                    env.passed_B

                    and

                    not B_passed_reported

                ):

                    print()

                    print(
                        "========== B PASSED =========="
                    )

                    print(
                        "A observation after B passed:",
                        obs_A
                    )

                    print(
                        "A other_passed feature:",
                        obs_A[6]
                    )

                    B_passed_reported = True


                # =============================================
                # UPDATE ACTOR HISTORIES
                # =============================================

                history_A.append(
                    obs_A.copy()
                )

                history_B.append(
                    obs_B.copy()
                )


                # =============================================
                # BUILD NEW CENTRALIZED STATE
                # =============================================

                state = build_centralized_state(

                    obs_A,

                    obs_B

                )


                # =============================================
                # UPDATE CRITIC HISTORY
                # =============================================

                state_history.append(
                    state.copy()
                )


                # =============================================
                # BOOTSTRAP VALUE FOR TIME-LIMIT TRUNCATION
                # =============================================

                bootstrap_values = np.zeros(
                    NUM_AGENTS,
                    dtype=np.float32
                )

                if truncated and not terminated:

                    next_state_sequence = history_to_array(
                        state_history
                    )

                    bootstrap_values = np.asarray(
                        mappo.get_value(
                            next_state_sequence
                        ),
                        dtype=np.float32
                    )


                # =============================================
                # STORE EXPERIENCE
                # =============================================

                buffer.add(

                    # IMPORTANT:
                    # Store the observation that produced action_t,
                    # not the observation returned by env.step().
                    observations=[

                        current_obs_A,

                        current_obs_B

                    ],

                    # IMPORTANT:
                    # Store state_t together with V(s_t).
                    # Do not store state_{t+1} with V(s_t).
                    state=current_state,

                    actions=[

                        np.array(
                            [action_A],
                            dtype=np.float32
                        ),

                        np.array(
                            [action_B],
                            dtype=np.float32
                        )

                    ],

                    rewards=[
                        reward_A,
                        reward_B
                    ],

                    log_probabilities=[
                        log_probabilities[0],
                        log_probabilities[1]
                    ],

                    values=np.asarray(
                        values,
                        dtype=np.float32
                    ),

                    terminated=terminated,
                    episode_end=episode_end,
                    bootstrap_values=bootstrap_values

                )


            # =================================================
            # EPISODE TERMINATED
            # =================================================

            collision = (

                env.collision_A

                or

                env.collision_B

            )


            # =================================================
            # BOTH VEHICLES PASSED
            # =================================================

            success = (

                env.passed_A

                and

                env.passed_B

            )


            # =================================================
            # TIME LIMIT
            # =================================================

            time_limit = bool(
                info.get(
                    "truncated",
                    False
                )
            )

            # Timeout without collision or joint success.
            deadlock = bool(
                (not collision)
                and
                (not success)
                and
                time_limit
            )


            # =================================================
            # UPDATE ROLLOUT COUNTER
            # =================================================

            episodes_in_rollout += 1


            # =================================================
            # CALCULATE AVERAGE TTC
            # =================================================

            if ttc_count > 0:

                average_ttc = (

                    ttc_sum /
                    ttc_count

                )

            else:

                average_ttc = 0.0


            # =================================================
            # SAVE EPISODE STATISTICS
            # =================================================

            episode_rewards_A.append(
                total_reward_A
            )

            episode_rewards_B.append(
                total_reward_B
            )

            episode_lengths.append(
                env.step_count
            )

            episode_collisions.append(
                collision
            )

            episode_successes.append(
                success
            )

            episode_time_limits.append(
                time_limit
            )

            episode_avg_ttc.append(
                average_ttc
            )


            # =================================================
            # DETERMINE WHETHER TO UPDATE
            # =================================================

            should_update = (

                episodes_in_rollout
                >= ROLLOUT_EPISODES

                or

                episode + 1
                >= NUM_EPISODES

            )


            # =================================================
            # DEFAULT LOSS VALUES
            # =================================================

            actor_loss = np.nan
            critic_loss = np.nan
            approx_kl = np.nan
            entropy_value = np.nan
            adv_mean = np.nan
            adv_std = np.nan
            value_mean = np.nan
            return_mean = np.nan


            # =================================================
            # MAPPO UPDATE
            # =================================================

            if should_update:

                print()

                print(
                    "======================================"
                )

                print(
                    "MAPPO PPO UPDATE"
                )

                print(
                    "======================================"
                )

                print(

                    f"Episodes collected: "
                    f"{episodes_in_rollout}"

                )

                print()

                # ---------------------------------------------
                # COMPUTE GAE
                # ---------------------------------------------

                # Every environment episode in this buffer has
                # already ended. True terminals use zero bootstrap;
                # time-limit truncations store V(next_state) inside
                # the buffer. last_values is only a fallback for
                # a rollout that ends mid-episode.

                last_values = np.array(

                    [

                        0.0,

                        0.0

                    ],

                    dtype=np.float32

                )


                buffer.compute_returns_and_advantages(

                    last_values=last_values,

                    gamma=GAMMA,

                    gae_lambda=GAE_LAMBDA

                )


                # ---------------------------------------------
                # UPDATE MAPPO
                # ---------------------------------------------

                update_result = (

                    mappo.update_from_buffer(

                        buffer,

                        epochs=MAPPO_EPOCHS

                    )

                )


                actor_loss = update_result.get(

                    "actor_loss",

                    np.nan

                )

                critic_loss = update_result.get(

                    "critic_loss",

                    np.nan

                )

                approx_kl = update_result.get("approx_kl", np.nan)
                entropy_value = update_result.get("entropy", np.nan)
                adv_mean = update_result.get("adv_mean", np.nan)
                adv_std = update_result.get("adv_std", np.nan)
                value_mean = update_result.get("value_mean", np.nan)
                return_mean = update_result.get("return_mean", np.nan)


                print(

                    f"Actor loss:  "
                    f"{actor_loss:.6f}"

                )

                print(

                    f"Critic loss: "
                    f"{critic_loss:.6f}"

                )

                print(
                    f"Approx KL:   {approx_kl:.6f}"
                )
                print(
                    f"Entropy:     {entropy_value:.6f}"
                )
                print(
                    f"Adv mean/std: "
                    f"{adv_mean:.6f} / {adv_std:.6f}"
                )
                print(
                    f"Value mean:  {value_mean:.6f}"
                )
                print(
                    f"Return mean: {return_mean:.6f}"
                )


                # ---------------------------------------------
                # RESET BUFFER
                # ---------------------------------------------

                buffer = create_rollout_buffer()

                episodes_in_rollout = 0


                print()

                print(
                    "Rollout buffer cleared."
                )

                print(
                    "======================================"
                )


            # =================================================
            # WRITE EPISODE TO CSV
            # =================================================

            with open(

                CSV_FILE,

                "a",

                newline=""

            ) as file:

                writer = csv.writer(file)

                writer.writerow([

                    episode + 1,

                    total_reward_A,

                    total_reward_B,

                    env.step_count,

                    collision,

                    env.passed_A,

                    env.passed_B,

                    success,

                    time_limit,

                    average_ttc,

                    actor_loss,

                    critic_loss,

                    approx_kl,

                    entropy_value,

                    adv_mean,

                    adv_std,

                    value_mean,

                    return_mean,

                    A_recovery_time_steps,

                    B_recovery_time_steps,

                    deadlock

                ])


            # =================================================
            # PRINT EPISODE SUMMARY
            # =================================================

            print()

            print(
                "--------------------------------------"
            )

            print(

                f"Episode "
                f"{episode + 1:3d}/"
                f"{NUM_EPISODES}"

            )

            print(
                "--------------------------------------"
            )

            print(

                f"Steps:       "
                f"{env.step_count}"

            )

            print(

                f"Reward A:    "
                f"{total_reward_A:8.3f}"

            )

            print(

                f"Reward B:    "
                f"{total_reward_B:8.3f}"

            )

            print(

                f"Collision:   "
                f"{collision}"

            )

            print(

                f"A passed:    "
                f"{env.passed_A}"

            )

            print(

                f"B passed:    "
                f"{env.passed_B}"

            )

            print(

                f"Success:     "
                f"{success}"

            )

            print(

                f"Time limit:  "
                f"{time_limit}"

            )

            if not np.isnan(actor_loss):

                print(

                    f"Actor loss:  "
                    f"{actor_loss:.6f}"

                )

                print(

                    f"Critic loss: "
                    f"{critic_loss:.6f}"

                )

            else:

                print(
                    "PPO update:  not yet"
                )


            if diagnostic_episode:

                print()
                print(
                    "========== TRAJECTORY DIAGNOSTICS =========="
                )
                print(
                    f"A stopped steps: {A_stopped_steps}"
                )
                print(
                    f"B stopped steps: {B_stopped_steps}"
                )
                print(
                    f"Saved to: {DIAGNOSTIC_CSV_FILE}"
                )
                print(
                    "============================================"
                )


            # =================================================
            # ROLLING 10-EPISODE STATISTICS
            # =================================================

            if (

                len(episode_rewards_A)
                >= ROLLING_WINDOW

            ):

                recent_A = (

                    episode_rewards_A[
                        -ROLLING_WINDOW:
                    ]

                )

                recent_B = (

                    episode_rewards_B[
                        -ROLLING_WINDOW:
                    ]

                )

                recent_collisions = (

                    episode_collisions[
                        -ROLLING_WINDOW:
                    ]

                )

                recent_successes = (

                    episode_successes[
                        -ROLLING_WINDOW:
                    ]

                )

                recent_lengths = (

                    episode_lengths[
                        -ROLLING_WINDOW:
                    ]

                )


                # =============================================
                # ROLLING AVERAGE REWARD
                # =============================================

                rolling_reward_A = np.mean(
                    recent_A
                )

                rolling_reward_B = np.mean(
                    recent_B
                )


                # =============================================
                # ROLLING COLLISION RATE
                # =============================================

                rolling_collision_rate = (

                    np.mean(
                        recent_collisions
                    ) * 100

                )


                # =============================================
                # ROLLING SUCCESS RATE
                # =============================================

                rolling_success_rate = (

                    np.mean(
                        recent_successes
                    ) * 100

                )


                # =============================================
                # ROLLING EPISODE LENGTH
                # =============================================

                rolling_episode_length = (

                    np.mean(
                        recent_lengths
                    )

                )


                # =============================================
                # PRINT ROLLING METRICS
                # =============================================

                print()

                print(
                    "========== ROLLING "
                    "10-EPISODE METRICS =========="
                )

                print(

                    f"Avg reward A:      "
                    f"{rolling_reward_A:.3f}"

                )

                print(

                    f"Avg reward B:      "
                    f"{rolling_reward_B:.3f}"

                )

                print(

                    f"Collision rate:    "
                    f"{rolling_collision_rate:.2f}%"

                )

                print(

                    f"Success rate:      "
                    f"{rolling_success_rate:.2f}%"

                )

                print(

                    f"Avg episode len:   "
                    f"{rolling_episode_length:.2f}"

                )

                print(
                    "======================================"
                )


    finally:

        # =====================================================
        # CLOSE CARLA
        # =====================================================

        env.close()


    # ========================================================
    # SAVE FINAL MAPPO MODEL
    # ========================================================

    torch.save(

        {

            "actor":
                mappo.actor.state_dict(),

            "critic":
                mappo.critic.state_dict()

        },

        MODEL_FILE

    )


    print()

    print(

        f"Final MAPPO model saved to: "
        f"{MODEL_FILE}"

    )


    # ========================================================
    # FINAL TRAINING STATISTICS
    # ========================================================

    total_episodes = len(
        episode_rewards_A
    )


    if total_episodes == 0:

        return


    collision_count = sum(
        episode_collisions
    )

    success_count = sum(
        episode_successes
    )

    time_limit_count = sum(
        episode_time_limits
    )


    # ========================================================
    # AVERAGE REWARDS
    # ========================================================

    average_reward_A = np.mean(
        episode_rewards_A
    )

    average_reward_B = np.mean(
        episode_rewards_B
    )


    # ========================================================
    # AVERAGE EPISODE LENGTH
    # ========================================================

    average_episode_length = np.mean(
        episode_lengths
    )


    # ========================================================
    # AVERAGE TTC
    # ========================================================

    average_ttc = np.mean(
        episode_avg_ttc
    )


    # ========================================================
    # COLLISION RATE
    # ========================================================

    collision_rate = (

        collision_count /
        total_episodes

    )


    # ========================================================
    # SUCCESS RATE
    # ========================================================

    success_rate = (

        success_count /
        total_episodes

    )


    # ========================================================
    # TIME LIMIT RATE
    # ========================================================

    time_limit_rate = (

        time_limit_count /
        total_episodes

    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print()

    print(
        "======================================"
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "======================================"
    )

    print()

    print(

        f"Total episodes:       "
        f"{total_episodes}"

    )

    print(

        f"Average reward A:     "
        f"{average_reward_A:.3f}"

    )

    print(

        f"Average reward B:     "
        f"{average_reward_B:.3f}"

    )

    print(

        f"Average episode:      "
        f"{average_episode_length:.2f} steps"

    )

    print(

        f"Average TTC:          "
        f"{average_ttc:.3f}"

    )

    print()

    print(

        f"Collisions:           "
        f"{collision_count}"

    )

    print(

        f"Collision rate:       "
        f"{collision_rate * 100:.2f}%"

    )

    print()

    print(

        f"Successful episodes:  "
        f"{success_count}"

    )

    print(

        f"Success rate:         "
        f"{success_rate * 100:.2f}%"

    )

    print()

    print(

        f"Time-limit episodes:  "
        f"{time_limit_count}"

    )

    print(

        f"Time-limit rate:      "
        f"{time_limit_rate * 100:.2f}%"

    )

    print()

    print(

        f"Training metrics:     "
        f"{CSV_FILE}"

    )

    print()

    print(
        f"Trajectory diagnostics: "
        f"{DIAGNOSTIC_CSV_FILE}"
    )

    print()

    print(

        f"Final MAPPO model:    "
        f"{MODEL_FILE}"

    )

    print()

    print(
        "======================================"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()




