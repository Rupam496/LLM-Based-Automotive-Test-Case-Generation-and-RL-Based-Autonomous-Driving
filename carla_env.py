import carla
import math
import random


# ============================================================
# CONFIGURATION
# ============================================================

CARLA_HOST = "localhost"
CARLA_PORT = 2000

MAP_NAME = "Town10HD_Opt"

JUNCTION_ID = 895

PAIR_A = 1
PAIR_B = 0

DT = 0.1

MAX_STEPS = 300

VEHICLE_MODEL = "vehicle.tesla.model3"

MIN_ACCELERATION = -3.0
MAX_ACCELERATION = 3.0

MAX_SPEED = 12.0


# ============================================================
# CONFLICT ZONE
# ============================================================

CONFLICT_ZONE_RADIUS = 8.0


# ============================================================
# VEHICLE SPAWN
# ============================================================

SPAWN_DISTANCE = 50.0


# ============================================================
# TRAJECTORY
# ============================================================

APPROACH_DISTANCE = 60.0

EXIT_DISTANCE = 40.0

TRAJECTORY_STEP = 2.0


# ============================================================
# TTC
# ============================================================

MIN_TTC_SPEED = 0.1

# Finite value used when TTC is undefined or very large.
# This prevents NaN/Inf from entering the neural network.

MAX_TTC = 100.0


# ============================================================
# REWARD PARAMETERS
# ============================================================

# Reward for moving toward the conflict point.
PROGRESS_WEIGHT = 1.0

# Comfort penalty.
COMFORT_WEIGHT = 0.01

# Penalty for unnecessary waiting.
WAITING_PENALTY = 0.05

# Stronger penalty for waiting after the other vehicle
# has already crossed.
POST_PASS_WAITING_PENALTY = 0.25

# Reward when one vehicle crosses the conflict point.
PASS_REWARD = 20.0

# Additional reward to BOTH vehicles when both have crossed.
JOINT_SUCCESS_REWARD = 40.0

# Collision penalty.
COLLISION_PENALTY = -100.0

# Penalty for unfinished vehicle at the time limit.
TIME_LIMIT_PENALTY = -20.0


# ============================================================
# TTC RISK PARAMETERS
# ============================================================

TTC_WARNING_THRESHOLD = 8.0

TTC_DIFFERENCE_THRESHOLD = 2.0

TTC_PENALTY_LEVEL_1 = 0.5
TTC_PENALTY_LEVEL_2 = 1.5
TTC_PENALTY_LEVEL_3 = 2.5
TTC_PENALTY_LEVEL_4 = 4.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance_2d(a, b):

    dx = a.x - b.x
    dy = a.y - b.y

    return math.sqrt(
        dx * dx +
        dy * dy
    )


def speed_of(vehicle):

    velocity = vehicle.get_velocity()

    return math.sqrt(
        velocity.x ** 2 +
        velocity.y ** 2 +
        velocity.z ** 2
    )


# ============================================================
# ENVIRONMENT
# ============================================================

class CarlaIntersectionEnv:

    def __init__(self):

        # ----------------------------------------------------
        # Connect to CARLA
        # ----------------------------------------------------

        self.client = carla.Client(
            CARLA_HOST,
            CARLA_PORT
        )

        self.client.set_timeout(
            10.0
        )

        self.world = self.client.get_world()

        print(
            "Map:",
            self.world.get_map().name
        )

        self.carla_map = (
            self.world.get_map()
        )


        # ----------------------------------------------------
        # Save original settings
        # ----------------------------------------------------

        self.original_settings = (
            self.world.get_settings()
        )


        # ----------------------------------------------------
        # Enable synchronous mode
        # ----------------------------------------------------

        settings = (
            self.world.get_settings()
        )

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = DT

        self.world.apply_settings(
            settings
        )

        print(
            "Synchronous mode enabled."
        )

        print(
            "Fixed timestep:",
            DT,
            "seconds"
        )


        # ----------------------------------------------------
        # Find junction
        # ----------------------------------------------------

        self.junction = (
            self.find_junction(
                JUNCTION_ID
            )
        )

        if self.junction is None:

            raise RuntimeError(
                f"Junction {JUNCTION_ID} not found."
            )

        print(
            "Junction found:",
            JUNCTION_ID
        )


        # ----------------------------------------------------
        # Get lane pairs
        # ----------------------------------------------------

        self.lane_pairs = (
            self.junction.get_waypoints(
                carla.LaneType.Driving
            )
        )

        print(
            "Number of lane pairs:",
            len(self.lane_pairs)
        )


        if len(self.lane_pairs) <= max(
            PAIR_A,
            PAIR_B
        ):

            raise RuntimeError(
                "Required lane pairs were not found."
            )


        # ----------------------------------------------------
        # Build trajectories
        # ----------------------------------------------------

        self.trajectory_A = (
            self.build_trajectory(
                self.lane_pairs[PAIR_A]
            )
        )

        self.trajectory_B = (
            self.build_trajectory(
                self.lane_pairs[PAIR_B]
            )
        )

        # ----------------------------------------------------
        # Store the two fixed physical approaches.
        #
        # These remain unchanged. During reset(), the two
        # approaches are randomly assigned to Agent A and
        # Agent B so that the agents do not learn a fixed
        # "A goes / B waits" role.
        # ----------------------------------------------------

        self.base_trajectory_A = self.trajectory_A
        self.base_trajectory_B = self.trajectory_B

        self.base_path_distance_A = (
            self.calculate_path_distances(
                self.base_trajectory_A
            )
        )

        self.base_path_distance_B = (
            self.calculate_path_distances(
                self.base_trajectory_B
            )
        )

        (
            self.base_conflict_point,
            self.base_conflict_index_A,
            self.base_conflict_index_B,
            self.base_trajectory_separation
        ) = self.find_conflict_point(
            self.base_trajectory_A,
            self.base_trajectory_B
        )

        self.base_conflict_path_distance_A = (
            self.base_path_distance_A[
                self.base_conflict_index_A
            ]
        )

        self.base_conflict_path_distance_B = (
            self.base_path_distance_B[
                self.base_conflict_index_B
            ]
        )

        # True means the physical approaches are swapped:
        # Agent A uses the original B approach and
        # Agent B uses the original A approach.
        self.roles_swapped = False


        print()

        print(
            "========== TRAJECTORIES =========="
        )

        print(
            "Trajectory A points:",
            len(self.trajectory_A)
        )

        print(
            "Trajectory B points:",
            len(self.trajectory_B)
        )


        # ----------------------------------------------------
        # Calculate path distances
        # ----------------------------------------------------

        self.path_distance_A = (
            self.calculate_path_distances(
                self.trajectory_A
            )
        )

        self.path_distance_B = (
            self.calculate_path_distances(
                self.trajectory_B
            )
        )


        # ----------------------------------------------------
        # Find conflict point
        # ----------------------------------------------------

        (
            self.conflict_point,
            self.conflict_index_A,
            self.conflict_index_B,
            self.trajectory_separation
        ) = self.find_conflict_point(

            self.trajectory_A,

            self.trajectory_B
        )


        # ----------------------------------------------------
        # Conflict path distances
        # ----------------------------------------------------

        self.conflict_path_distance_A = (

            self.path_distance_A[
                self.conflict_index_A
            ]

        )

        self.conflict_path_distance_B = (

            self.path_distance_B[
                self.conflict_index_B
            ]

        )


        print()

        print(
            "========== CONFLICT POINT =========="
        )

        print(
            f"x = {self.conflict_point.x:.2f}"
        )

        print(
            f"y = {self.conflict_point.y:.2f}"
        )

        print(
            f"z = {self.conflict_point.z:.2f}"
        )

        print(
            "Trajectory separation:",
            f"{self.trajectory_separation:.2f} m"
        )

        print(
            "Car A conflict index:",
            self.conflict_index_A
        )

        print(
            "Car B conflict index:",
            self.conflict_index_B
        )


        # ----------------------------------------------------
        # Vehicles and sensors
        # ----------------------------------------------------

        self.vehicle_A = None

        self.vehicle_B = None

        self.collision_sensor_A = None

        self.collision_sensor_B = None


        # ----------------------------------------------------
        # Episode state
        # ----------------------------------------------------

        self.collision_A = False

        self.collision_B = False

        self.passed_A = False

        self.passed_B = False

        self.previous_distance_A = None

        self.previous_distance_B = None

        self.step_count = 0


        # ----------------------------------------------------
        # V2X state
        # ----------------------------------------------------

        self.v2x_A = None

        self.v2x_B = None


# ============================================================
# FIND JUNCTION
# ============================================================

    def find_junction(
        self,
        junction_id
    ):

        waypoints = (
            self.carla_map.generate_waypoints(
                2.0
            )
        )

        for waypoint in waypoints:

            if waypoint.is_junction:

                junction = (
                    waypoint.get_junction()
                )

                if junction.id == junction_id:

                    return junction

        return None


# ============================================================
# BUILD TRAJECTORY
# ============================================================

    def build_trajectory(
        self,
        lane_pair
    ):

        start_wp, end_wp = lane_pair


        # ----------------------------------------------------
        # Approach section
        # ----------------------------------------------------

        approach = []

        current_wp = start_wp

        for _ in range(
            int(
                APPROACH_DISTANCE /
                TRAJECTORY_STEP
            )
        ):

            approach.append(
                current_wp
            )

            previous_wps = (
                current_wp.previous(
                    TRAJECTORY_STEP
                )
            )

            if not previous_wps:

                break

            current_wp = previous_wps[0]


        approach.reverse()

        trajectory = approach


        # ----------------------------------------------------
        # Junction section
        # ----------------------------------------------------

        current_wp = start_wp

        while True:

            if (

                len(trajectory) == 0

                or

                trajectory[-1]
                .transform
                .location
                .distance(
                    current_wp.transform.location
                ) > 0.01

            ):

                trajectory.append(
                    current_wp
                )


            if (

                distance_2d(

                    current_wp.transform.location,

                    end_wp.transform.location

                )

                <

                TRAJECTORY_STEP

            ):

                break


            next_wps = (
                current_wp.next(
                    TRAJECTORY_STEP
                )
            )


            if not next_wps:

                break


            current_wp = next_wps[0]


            if len(trajectory) > 500:

                break


        # ----------------------------------------------------
        # Exit section
        # ----------------------------------------------------

        current_wp = trajectory[-1]

        for _ in range(
            int(
                EXIT_DISTANCE /
                TRAJECTORY_STEP
            )
        ):

            next_wps = (
                current_wp.next(
                    TRAJECTORY_STEP
                )
            )


            if not next_wps:

                break


            current_wp = next_wps[0]

            trajectory.append(
                current_wp
            )


        return trajectory


# ============================================================
# FIND CONFLICT POINT
# ============================================================

    def find_conflict_point(
        self,
        trajectory_A,
        trajectory_B
    ):

        minimum_distance = float("inf")

        best_i = None

        best_j = None


        for i, wp_A in enumerate(
            trajectory_A
        ):

            location_A = (
                wp_A.transform.location
            )


            for j, wp_B in enumerate(
                trajectory_B
            ):

                location_B = (
                    wp_B.transform.location
                )


                distance = distance_2d(

                    location_A,

                    location_B

                )


                if distance < minimum_distance:

                    minimum_distance = distance

                    best_i = i

                    best_j = j


        location_A = (

            trajectory_A[
                best_i
            ].transform.location

        )


        location_B = (

            trajectory_B[
                best_j
            ].transform.location

        )


        conflict_point = carla.Location(

            x=(

                location_A.x +
                location_B.x

            ) / 2.0,

            y=(

                location_A.y +
                location_B.y

            ) / 2.0,

            z=(

                location_A.z +
                location_B.z

            ) / 2.0

        )


        return (

            conflict_point,

            best_i,

            best_j,

            minimum_distance

        )


# ============================================================
# PATH DISTANCES
# ============================================================

    def calculate_path_distances(
        self,
        trajectory
    ):

        distances = [0.0]


        for i in range(
            1,
            len(trajectory)
        ):

            d = distance_2d(

                trajectory[
                    i - 1
                ].transform.location,

                trajectory[
                    i
                ].transform.location

            )


            distances.append(

                distances[-1] + d

            )


        return distances


# ============================================================
# PROJECT VEHICLE ONTO TRAJECTORY
# ============================================================

    def project_vehicle_onto_trajectory(
        self,
        vehicle,
        trajectory,
        path_distances,
        conflict_path_distance
    ):

        vehicle_location = (
            vehicle.get_location()
        )


        best_distance = float("inf")

        best_path_position = None


        for i in range(
            len(trajectory) - 1
        ):

            p1 = (

                trajectory[
                    i
                ].transform.location

            )


            p2 = (

                trajectory[
                    i + 1
                ].transform.location

            )


            vx = p2.x - p1.x

            vy = p2.y - p1.y


            segment_length_squared = (

                vx * vx +
                vy * vy

            )


            if segment_length_squared == 0:

                continue


            wx = (

                vehicle_location.x -
                p1.x

            )


            wy = (

                vehicle_location.y -
                p1.y

            )


            t = (

                wx * vx +
                wy * vy

            ) / segment_length_squared


            t = max(

                0.0,

                min(
                    1.0,
                    t
                )

            )


            projected_x = (

                p1.x +
                t * vx

            )


            projected_y = (

                p1.y +
                t * vy

            )


            dx = (

                vehicle_location.x -
                projected_x

            )


            dy = (

                vehicle_location.y -
                projected_y

            )


            distance_to_segment = (

                dx * dx +
                dy * dy

            )


            if (

                distance_to_segment
                <
                best_distance

            ):

                best_distance = (
                    distance_to_segment
                )


                segment_length = math.sqrt(

                    segment_length_squared

                )


                best_path_position = (

                    path_distances[i]

                    +

                    t *
                    segment_length

                )


        if best_path_position is None:

            raise RuntimeError(

                "Could not project vehicle "
                "onto trajectory."

            )


        # Positive = before conflict
        # Zero     = conflict point
        # Negative = after conflict

        signed_distance = (

            conflict_path_distance
            -
            best_path_position

        )


        return signed_distance


# ============================================================
# TTC
# ============================================================

    def calculate_ttc(
        self,
        distance_to_conflict,
        speed
    ):

        # Already at/past conflict.
        if distance_to_conflict <= 0:

            return MAX_TTC


        # Vehicle almost stopped.
        if speed < MIN_TTC_SPEED:

            return MAX_TTC


        ttc = (

            distance_to_conflict /
            speed

        )


        if not math.isfinite(ttc):

            return MAX_TTC


        return min(
            ttc,
            MAX_TTC
        )


# ============================================================
# SPAWN VEHICLE
# ============================================================

    def spawn_vehicle(
        self,
        trajectory,
        path_distances,
        conflict_path_distance
    ):

        target_path_distance = (

            conflict_path_distance
            -
            SPAWN_DISTANCE

        )


        spawn_index = min(

            range(
                len(trajectory)
            ),

            key=lambda i:

                abs(

                    path_distances[i]
                    -
                    target_path_distance

                )

        )


        location = (
            trajectory[
                spawn_index
            ]
            .transform
            .location
        )


        rotation = (
            trajectory[
                spawn_index
            ]
            .transform
            .rotation
        )


        transform = carla.Transform(

            carla.Location(

                x=location.x,

                y=location.y,

                z=location.z + 0.15

            ),

            carla.Rotation(

                pitch=rotation.pitch,

                yaw=rotation.yaw,

                roll=rotation.roll

            )

        )


        blueprint = (

            self.world
            .get_blueprint_library()
            .find(
                VEHICLE_MODEL
            )

        )


        vehicle = (

            self.world.try_spawn_actor(

                blueprint,

                transform

            )

        )


        if vehicle is None:

            raise RuntimeError(
                "Vehicle spawn failed."
            )


        return (

            vehicle,

            spawn_index

        )


# ============================================================
# COLLISION CALLBACK A
# ============================================================

    def collision_callback_A(
        self,
        event
    ):

        other_actor = (
            event.other_actor
        )


        if (

            other_actor is not None

            and

            self.vehicle_B is not None

            and

            other_actor.id
            ==
            self.vehicle_B.id

        ):

            self.collision_A = True

            print()

            print(
                "!!! A <-> B COLLISION DETECTED !!!"
            )


# ============================================================
# COLLISION CALLBACK B
# ============================================================

    def collision_callback_B(
        self,
        event
    ):

        other_actor = (
            event.other_actor
        )


        if (

            other_actor is not None

            and

            self.vehicle_A is not None

            and

            other_actor.id
            ==
            self.vehicle_A.id

        ):

            self.collision_B = True


# ============================================================
# V2X EXCHANGE
# ============================================================

    def exchange_v2x(self):

        distance_A = (

            self.project_vehicle_onto_trajectory(

                self.vehicle_A,

                self.trajectory_A,

                self.path_distance_A,

                self.conflict_path_distance_A

            )

        )


        distance_B = (

            self.project_vehicle_onto_trajectory(

                self.vehicle_B,

                self.trajectory_B,

                self.path_distance_B,

                self.conflict_path_distance_B

            )

        )


        speed_A = speed_of(
            self.vehicle_A
        )


        speed_B = speed_of(
            self.vehicle_B
        )


        ttc_A = self.calculate_ttc(

            distance_A,

            speed_A

        )


        ttc_B = self.calculate_ttc(

            distance_B,

            speed_B

        )


        # ----------------------------------------------------
        # Vehicle A's transmitted information
        # ----------------------------------------------------

        self.v2x_A = {

            "distance_to_conflict":
                distance_A,

            "speed":
                speed_A,

            "ttc":
                ttc_A

        }


        # ----------------------------------------------------
        # Vehicle B's transmitted information
        # ----------------------------------------------------

        self.v2x_B = {

            "distance_to_conflict":
                distance_B,

            "speed":
                speed_B,

            "ttc":
                ttc_B

        }


# ============================================================
# OBSERVATIONS
# ============================================================

    def get_observations(self):

        distance_A = (

            self.project_vehicle_onto_trajectory(

                self.vehicle_A,

                self.trajectory_A,

                self.path_distance_A,

                self.conflict_path_distance_A

            )

        )


        distance_B = (

            self.project_vehicle_onto_trajectory(

                self.vehicle_B,

                self.trajectory_B,

                self.path_distance_B,

                self.conflict_path_distance_B

            )

        )


        speed_A = speed_of(
            self.vehicle_A
        )


        speed_B = speed_of(
            self.vehicle_B
        )


        ttc_A = self.calculate_ttc(

            distance_A,

            speed_A

        )


        ttc_B = self.calculate_ttc(

            distance_B,

            speed_B

        )


        # ----------------------------------------------------
        # Conflict zone
        # ----------------------------------------------------

        in_zone_A = (

            abs(distance_A)
            <=
            CONFLICT_ZONE_RADIUS

        )


        in_zone_B = (

            abs(distance_B)
            <=
            CONFLICT_ZONE_RADIUS

        )


        # ====================================================
        # Agent A observation
        # ====================================================
        #
        # 0: own distance to conflict
        # 1: own speed
        # 2: own TTC
        # 3: other vehicle distance to conflict
        # 4: other vehicle speed
        # 5: other vehicle TTC
        # 6: other vehicle passed
        #
        # This is explicitly SELF / OTHER structured.
        # ====================================================

        obs_A = [

            distance_A,

            speed_A,

            ttc_A,

            distance_B,

            speed_B,

            ttc_B,

            float(self.passed_B)

        ]


        # ====================================================
        # Agent B observation
        # ====================================================
        #
        # 0: own distance to conflict
        # 1: own speed
        # 2: own TTC
        # 3: other vehicle distance to conflict
        # 4: other vehicle speed
        # 5: other vehicle TTC
        # 6: other vehicle passed
        #
        # Agent B sees the same semantic feature ordering
        # as Agent A. Only the physical vehicle identities
        # change.
        # ====================================================

        obs_B = [

            distance_B,

            speed_B,

            ttc_B,

            distance_A,

            speed_A,

            ttc_A,

            float(self.passed_A)

        ]


        # ----------------------------------------------------
        # Numerical safety
        # ----------------------------------------------------

        if not all(

            math.isfinite(
                float(x)
            )

            for x in obs_A

        ):

            raise RuntimeError(

                "Observation A contains "
                "NaN or Inf."

            )


        if not all(

            math.isfinite(
                float(x)
            )

            for x in obs_B

        ):

            raise RuntimeError(

                "Observation B contains "
                "NaN or Inf."

            )


        return (

            obs_A,

            obs_B,

            in_zone_A,

            in_zone_B,

            ttc_A,

            ttc_B

        )


# ============================================================
# LOW LEVEL CONTROL
# ============================================================

    def apply_action(
        self,
        vehicle,
        desired_acceleration
    ):

        desired_acceleration = max(

            MIN_ACCELERATION,

            min(

                MAX_ACCELERATION,

                float(
                    desired_acceleration
                )

            )

        )


        current_speed = speed_of(
            vehicle
        )


        # ----------------------------------------------------
        # Speed limit
        # ----------------------------------------------------

        if current_speed >= MAX_SPEED:

            desired_acceleration = min(

                desired_acceleration,

                0.0

            )


        # ----------------------------------------------------
        # Positive acceleration
        # ----------------------------------------------------

        if desired_acceleration > 0:

            throttle = (

                0.15
                +
                0.20 *
                desired_acceleration

            )


            throttle = min(

                throttle,

                1.0

            )


            brake = 0.0


        # ----------------------------------------------------
        # Negative acceleration
        # ----------------------------------------------------

        elif desired_acceleration < 0:

            throttle = 0.0


            brake = min(

                0.25 *
                abs(
                    desired_acceleration
                ),

                1.0

            )


        # ----------------------------------------------------
        # Zero acceleration
        # ----------------------------------------------------

        else:

            throttle = 0.15

            brake = 0.0


        control = carla.VehicleControl()

        control.throttle = throttle

        control.brake = brake

        control.steer = 0.0


        vehicle.apply_control(
            control
        )


# ============================================================
# STOP VEHICLE
# ============================================================

    def stop_vehicle(
        self,
        vehicle
    ):

        control = carla.VehicleControl()

        control.throttle = 0.0

        control.brake = 1.0

        control.steer = 0.0


        vehicle.apply_control(
            control
        )


# ============================================================
# TTC RISK
# ============================================================

    def calculate_ttc_risk(
        self,
        ttc_A,
        ttc_B
    ):

        # ----------------------------------------------------
        # If one vehicle has already crossed, there is no
        # longer an active conflict between the two approaches.
        # ----------------------------------------------------

        if self.passed_A or self.passed_B:

            return False, 0.0


        # ----------------------------------------------------
        # Both vehicles must have TTC below warning threshold.
        # ----------------------------------------------------

        if ttc_A >= TTC_WARNING_THRESHOLD:

            return False, 0.0


        if ttc_B >= TTC_WARNING_THRESHOLD:

            return False, 0.0


        # ----------------------------------------------------
        # TTC values must be reasonably close.
        # ----------------------------------------------------

        ttc_difference = abs(

            ttc_A -
            ttc_B

        )


        if (

            ttc_difference
            >
            TTC_DIFFERENCE_THRESHOLD

        ):

            return False, 0.0


        # ----------------------------------------------------
        # Progressive penalty
        # ----------------------------------------------------

        minimum_ttc = min(

            ttc_A,

            ttc_B

        )


        if minimum_ttc >= 6.0:

            penalty = (
                TTC_PENALTY_LEVEL_1
            )

        elif minimum_ttc >= 5.0:

            penalty = (
                TTC_PENALTY_LEVEL_2
            )

        elif minimum_ttc >= 4.0:

            penalty = (
                TTC_PENALTY_LEVEL_3
            )

        else:

            penalty = (
                TTC_PENALTY_LEVEL_4
            )


        return True, penalty


# ============================================================
# REWARD
# ============================================================

    def calculate_reward(
        self,
        previous_distance,
        current_distance,
        current_speed,
        action,
        newly_passed,
        other_passed,
        risky_interaction,
        risk_penalty,
        time_limit,
        already_passed
    ):

        # ----------------------------------------------------
        # Collision
        # ----------------------------------------------------

        if (

            self.collision_A

            or

            self.collision_B

        ):

            return COLLISION_PENALTY


        # ----------------------------------------------------
        # Already crossed in an earlier timestep
        # ----------------------------------------------------

        if already_passed:

            return 0.0


        # ----------------------------------------------------
        # Progress reward
        # ----------------------------------------------------

        progress = (
            previous_distance
            -
            current_distance
        )

        # After the other vehicle has safely crossed, reward
        # progress slightly more strongly. This encourages the
        # remaining vehicle to recover from unnecessary waiting.
        # It does not directly force an acceleration action.
        progress_weight = PROGRESS_WEIGHT

        if other_passed and not already_passed:
            progress_weight *= 1.5

        reward = (
            progress_weight *
            progress
        )


        # ----------------------------------------------------
        # Comfort penalty
        # ----------------------------------------------------

        reward -= (

            COMFORT_WEIGHT *
            float(action) ** 2

        )


        # ----------------------------------------------------
        # TTC risk penalty
        # ----------------------------------------------------

        if risky_interaction:

            reward -= risk_penalty


        # ----------------------------------------------------
        # WAITING PENALTY
        # ----------------------------------------------------

        if (

            not other_passed

            and

            not risky_interaction

            and

            current_distance > 2.0

            and

            current_speed < 0.5

        ):

            reward -= WAITING_PENALTY


        # ----------------------------------------------------
        # POST-PASS WAITING PENALTY
        # ----------------------------------------------------

        if (

            other_passed

            and

            current_distance > 2.0

            and

            current_speed < 0.5

        ):

            reward -= POST_PASS_WAITING_PENALTY


        # ----------------------------------------------------
        # Individual crossing reward
        # ----------------------------------------------------

        if newly_passed:

            reward += PASS_REWARD


        # ----------------------------------------------------
        # Time-limit penalty
        # ----------------------------------------------------

        if time_limit:

            reward += TIME_LIMIT_PENALTY


        return reward


# ============================================================
# RESET
# ============================================================

    def reset(self):

        # ----------------------------------------------------
        # Destroy old sensors
        # ----------------------------------------------------

        if self.collision_sensor_A:

            self.collision_sensor_A.destroy()

            self.collision_sensor_A = None


        if self.collision_sensor_B:

            self.collision_sensor_B.destroy()

            self.collision_sensor_B = None


        # ----------------------------------------------------
        # Destroy old vehicles
        # ----------------------------------------------------

        if self.vehicle_A:

            self.vehicle_A.destroy()

            self.vehicle_A = None


        if self.vehicle_B:

            self.vehicle_B.destroy()

            self.vehicle_B = None


        # ----------------------------------------------------
        # Reset state
        # ----------------------------------------------------

        self.collision_A = False

        self.collision_B = False

        self.passed_A = False

        self.passed_B = False

        self.step_count = 0

        self.v2x_A = None

        self.v2x_B = None

        self.previous_distance_A = None

        self.previous_distance_B = None


        # ----------------------------------------------------
        # Randomize the physical approach assigned to each
        # agent for this episode.
        #
        # Case 1:
        #   Agent A -> original A approach
        #   Agent B -> original B approach
        #
        # Case 2:
        #   Agent A -> original B approach
        #   Agent B -> original A approach
        #
        # This prevents the policy from simply associating
        # the fixed agent identity "A" with going first and
        # "B" with waiting.
        # ----------------------------------------------------

        self.roles_swapped = random.choice(
            [False, True]
        )

        if not self.roles_swapped:

            self.trajectory_A = (
                self.base_trajectory_A
            )

            self.trajectory_B = (
                self.base_trajectory_B
            )

            self.path_distance_A = (
                self.base_path_distance_A
            )

            self.path_distance_B = (
                self.base_path_distance_B
            )

            self.conflict_path_distance_A = (
                self.base_conflict_path_distance_A
            )

            self.conflict_path_distance_B = (
                self.base_conflict_path_distance_B
            )

            conflict_index_A = (
                self.base_conflict_index_A
            )

            conflict_index_B = (
                self.base_conflict_index_B
            )

        else:

            self.trajectory_A = (
                self.base_trajectory_B
            )

            self.trajectory_B = (
                self.base_trajectory_A
            )

            self.path_distance_A = (
                self.base_path_distance_B
            )

            self.path_distance_B = (
                self.base_path_distance_A
            )

            self.conflict_path_distance_A = (
                self.base_conflict_path_distance_B
            )

            self.conflict_path_distance_B = (
                self.base_conflict_path_distance_A
            )

            conflict_index_A = (
                self.base_conflict_index_B
            )

            conflict_index_B = (
                self.base_conflict_index_A
            )


        # ----------------------------------------------------
        # Spawn Agent A
        # ----------------------------------------------------

        (

            self.vehicle_A,

            index_A

        ) = self.spawn_vehicle(

            self.trajectory_A,

            self.path_distance_A,

            self.conflict_path_distance_A

        )


        # ----------------------------------------------------
        # Spawn Agent B
        # ----------------------------------------------------

        (

            self.vehicle_B,

            index_B

        ) = self.spawn_vehicle(

            self.trajectory_B,

            self.path_distance_B,

            self.conflict_path_distance_B

        )


        print()

        print(
            "========== EPISODE ROLE ASSIGNMENT =========="
        )

        if self.roles_swapped:

            print(
                "Agent A -> original B approach"
            )

            print(
                "Agent B -> original A approach"
            )

        else:

            print(
                "Agent A -> original A approach"
            )

            print(
                "Agent B -> original B approach"
            )


        print()

        print(
            "Car A spawned."
        )

        print(
            "Trajectory index:",
            index_A
        )


        print()

        print(
            "Car B spawned."
        )

        print(
            "Trajectory index:",
            index_B
        )


        # ----------------------------------------------------
        # Collision sensor
        # ----------------------------------------------------

        collision_bp = (

            self.world
            .get_blueprint_library()
            .find(
                "sensor.other.collision"
            )

        )


        self.collision_sensor_A = (

            self.world.spawn_actor(

                collision_bp,

                carla.Transform(),

                attach_to=self.vehicle_A

            )

        )


        self.collision_sensor_B = (

            self.world.spawn_actor(

                collision_bp,

                carla.Transform(),

                attach_to=self.vehicle_B

            )

        )


        self.collision_sensor_A.listen(

            self.collision_callback_A

        )


        self.collision_sensor_B.listen(

            self.collision_callback_B

        )


        # ----------------------------------------------------
        # Initial tick
        # ----------------------------------------------------

        self.world.tick()


        # ----------------------------------------------------
        # Initial observation
        # ----------------------------------------------------

        (

            obs_A,

            obs_B,

            in_zone_A,

            in_zone_B,

            ttc_A,

            ttc_B

        ) = self.get_observations()


        # ----------------------------------------------------
        # Previous distances
        # ----------------------------------------------------

        self.previous_distance_A = (
            obs_A[0]
        )

        self.previous_distance_B = (
            obs_B[0]
        )


        # ----------------------------------------------------
        # Initial V2X exchange
        # ----------------------------------------------------

        self.exchange_v2x()


        print()

        print(
            "======================================"
        )

        print(
            "RESET COMPLETE"
        )

        print(
            "======================================"
        )

        print(
            f"Car A distance: "
            f"{obs_A[0]:.2f} m"
        )

        print(
            f"Car B distance: "
            f"{obs_B[0]:.2f} m"
        )

        print(
            f"Car A TTC: "
            f"{ttc_A:.2f} s"
        )

        print(
            f"Car B TTC: "
            f"{ttc_B:.2f} s"
        )


        # ----------------------------------------------------
        # Initial vehicle state
        # ----------------------------------------------------

        print()

        print(
            "========== INITIAL VEHICLE STATE =========="
        )


        location_A = (
            self.vehicle_A
            .get_location()
        )

        location_B = (
            self.vehicle_B
            .get_location()
        )


        print(

            "Car A initial position: "

            f"x={location_A.x:.3f}, "

            f"y={location_A.y:.3f}, "

            f"z={location_A.z:.3f}"

        )


        print(

            "Car A initial speed: "

            f"{speed_of(self.vehicle_A):.3f} m/s"

        )


        print(

            "Car B initial position: "

            f"x={location_B.x:.3f}, "

            f"y={location_B.y:.3f}, "

            f"z={location_B.z:.3f}"

        )


        print(

            "Car B initial speed: "

            f"{speed_of(self.vehicle_B):.3f} m/s"

        )


        print()

        print(
            "Initial observation A:",
            obs_A
        )

        print(
            "Initial observation B:",
            obs_B
        )


        # IMPORTANT:
        #
        # Keep exactly two returned values because the current
        # MAPPO training code uses:
        #
        # obs_A, obs_B = env.reset()

        return (

            obs_A,

            obs_B

        )


# ============================================================
# STEP
# ============================================================

    def step(
        self,
        action_A,
        action_B
    ):

        # ----------------------------------------------------
        # Apply action A
        # ----------------------------------------------------

        if not self.passed_A:

            self.apply_action(

                self.vehicle_A,

                action_A

            )

        else:

            self.stop_vehicle(

                self.vehicle_A

            )


        # ----------------------------------------------------
        # Apply action B
        # ----------------------------------------------------

        if not self.passed_B:

            self.apply_action(

                self.vehicle_B,

                action_B

            )

        else:

            self.stop_vehicle(

                self.vehicle_B

            )


        # ----------------------------------------------------
        # Advance CARLA
        # ----------------------------------------------------

        self.world.tick()

        self.step_count += 1


        # ----------------------------------------------------
        # Current distances
        # ----------------------------------------------------

        current_distance_A = (

            self.project_vehicle_onto_trajectory(

                self.vehicle_A,

                self.trajectory_A,

                self.path_distance_A,

                self.conflict_path_distance_A

            )

        )


        current_distance_B = (

            self.project_vehicle_onto_trajectory(

                self.vehicle_B,

                self.trajectory_B,

                self.path_distance_B,

                self.conflict_path_distance_B

            )

        )


        # ----------------------------------------------------
        # Current speeds
        # ----------------------------------------------------

        speed_A = speed_of(
            self.vehicle_A
        )

        speed_B = speed_of(
            self.vehicle_B
        )


        # ----------------------------------------------------
        # TTC
        # ----------------------------------------------------

        ttc_A = self.calculate_ttc(

            current_distance_A,

            speed_A

        )


        ttc_B = self.calculate_ttc(

            current_distance_B,

            speed_B

        )


        # ----------------------------------------------------
        # Conflict-zone state
        # ----------------------------------------------------

        in_zone_A = (

            abs(current_distance_A)
            <=
            CONFLICT_ZONE_RADIUS

        )


        in_zone_B = (

            abs(current_distance_B)
            <=
            CONFLICT_ZONE_RADIUS

        )


        # ----------------------------------------------------
        # Calculate risk BEFORE changing passed flags
        # ----------------------------------------------------

        risky_interaction, risk_penalty = (

            self.calculate_ttc_risk(

                ttc_A,

                ttc_B

            )

        )


        ttc_difference = abs(

            ttc_A -
            ttc_B

        )


        # ----------------------------------------------------
        # Detect Car A passing
        # ----------------------------------------------------

        newly_passed_A = False


        if (

            not self.passed_A

            and

            self.previous_distance_A > 0.0

            and

            current_distance_A <= 0.0

        ):

            self.passed_A = True

            newly_passed_A = True


            self.stop_vehicle(

                self.vehicle_A

            )


        # ----------------------------------------------------
        # Detect Car B passing
        # ----------------------------------------------------

        newly_passed_B = False


        if (

            not self.passed_B

            and

            self.previous_distance_B > 0.0

            and

            current_distance_B <= 0.0

        ):

            self.passed_B = True

            newly_passed_B = True


            self.stop_vehicle(

                self.vehicle_B

            )


        # ----------------------------------------------------
        # Collision
        # ----------------------------------------------------

        collision = (

            self.collision_A
            or
            self.collision_B

        )


        # ----------------------------------------------------
        # Joint success
        # ----------------------------------------------------

        both_passed = (

            self.passed_A
            and
            self.passed_B

        )


        # ----------------------------------------------------
        # Time limit
        # ----------------------------------------------------

        time_limit = (

            self.step_count
            >=
            MAX_STEPS

        )


        # ----------------------------------------------------
        # Reward A
        # ----------------------------------------------------

        reward_A = self.calculate_reward(

            self.previous_distance_A,

            current_distance_A,

            speed_A,

            action_A,

            newly_passed_A,

            other_passed=self.passed_B,

            risky_interaction=risky_interaction,

            risk_penalty=risk_penalty,

            time_limit=(

                time_limit
                and
                not self.passed_A

            ),

            already_passed=(

                self.passed_A
                and
                not newly_passed_A

            )

        )


        # ----------------------------------------------------
        # Reward B
        # ----------------------------------------------------

        reward_B = self.calculate_reward(

            self.previous_distance_B,

            current_distance_B,

            speed_B,

            action_B,

            newly_passed_B,

            other_passed=self.passed_A,

            risky_interaction=risky_interaction,

            risk_penalty=risk_penalty,

            time_limit=(

                time_limit
                and
                not self.passed_B

            ),

            already_passed=(

                self.passed_B
                and
                not newly_passed_B

            )

        )


        # ----------------------------------------------------
        # Joint success reward
        # ----------------------------------------------------

        if (

            both_passed

            and

            (
                newly_passed_A
                or
                newly_passed_B
            )

        ):

            reward_A += (
                JOINT_SUCCESS_REWARD
            )

            reward_B += (
                JOINT_SUCCESS_REWARD
            )


        # ----------------------------------------------------
        # Update previous distances
        # ----------------------------------------------------

        self.previous_distance_A = (

            current_distance_A

        )

        self.previous_distance_B = (

            current_distance_B

        )


        # ----------------------------------------------------
        # V2X exchange
        # ----------------------------------------------------

        self.exchange_v2x()


        # ----------------------------------------------------
        # Final observations
        # ----------------------------------------------------

        (

            obs_A,

            obs_B,

            in_zone_A,

            in_zone_B,

            ttc_A,

            ttc_B

        ) = self.get_observations()


        # ----------------------------------------------------
        # Episode termination
        # ----------------------------------------------------
        #
        # terminated:
        #   A true terminal condition caused by the environment:
        #   collision or successful completion.
        #
        # truncated:
        #   The episode was stopped because MAX_STEPS was reached.
        #
        # This distinction is important for MAPPO/PPO:
        # a time-limit transition should bootstrap from the
        # value of the next state, whereas a true terminal
        # transition should not.
        # ----------------------------------------------------

        terminated = (

            collision

            or

            both_passed

        )

        truncated = (

            time_limit

            and

            not terminated

        )

        done = (

            terminated

            or

            truncated

        )


        # ----------------------------------------------------
        # Info
        # ----------------------------------------------------

        info = {

            "step":
                self.step_count,

            "collision_A":
                self.collision_A,

            "collision_B":
                self.collision_B,

            "passed_A":
                self.passed_A,

            "passed_B":
                self.passed_B,

            "newly_passed_A":
                newly_passed_A,

            "newly_passed_B":
                newly_passed_B,

            # Authoritative event information for diagnostics.
            # The training script records the actual timestep,
            # while these flags identify the exact transition in
            # which the crossing occurred.
            "pass_event_A":
                newly_passed_A,

            "pass_event_B":
                newly_passed_B,

            "distance_A":
                current_distance_A,

            "distance_B":
                current_distance_B,

            "speed_A":
                speed_A,

            "speed_B":
                speed_B,

            "ttc_A":
                ttc_A,

            "ttc_B":
                ttc_B,

            "ttc_difference":
                ttc_difference,

            "risky_interaction":
                risky_interaction,

            "risk_penalty":
                risk_penalty,

            "in_conflict_zone_A":
                in_zone_A,

            "in_conflict_zone_B":
                in_zone_B,

            "time_limit":
                time_limit,

            "terminated":
                terminated,

            "truncated":
                truncated,

            "both_passed":
                both_passed,

            "success":
                both_passed,

            "v2x_A":
                self.v2x_A,

            "v2x_B":
                self.v2x_B

        }


        return (

            obs_A,

            obs_B,

            reward_A,

            reward_B,

            done,

            info

        )


# ============================================================
# CLOSE
# ============================================================

    def close(self):

        print()

        print(
            "Destroying collision sensors..."
        )


        if self.collision_sensor_A:

            self.collision_sensor_A.destroy()

            self.collision_sensor_A = None


        if self.collision_sensor_B:

            self.collision_sensor_B.destroy()

            self.collision_sensor_B = None


        print(
            "Destroying vehicles..."
        )


        if self.vehicle_A:

            self.vehicle_A.destroy()

            self.vehicle_A = None


        if self.vehicle_B:

            self.vehicle_B.destroy()

            self.vehicle_B = None


        # ----------------------------------------------------
        # Restore original CARLA settings
        # ----------------------------------------------------

        self.world.apply_settings(

            self.original_settings

        )


        print(
            "CARLA settings restored."
        )

        print(
            "Environment closed."
        )


# ============================================================
# DIAGNOSTIC TEST
# ============================================================

if __name__ == "__main__":

    env = CarlaIntersectionEnv()

    try:

        obs_A, obs_B = env.reset()


        print()

        print(
            "======================================"
        )

        print(
            "STARTING FULL DRIVING TEST"
        )

        print(
            "======================================"
        )

        print(
            "Car A action = +1.0 m/s^2"
        )

        print(
            "Car B action = +1.0 m/s^2"
        )

        print(
            f"Conflict zone radius = "
            f"{CONFLICT_ZONE_RADIUS} m"
        )

        print(
            f"Maximum steps = "
            f"{MAX_STEPS}"
        )

        print()


        while True:

            (

                obs_A,

                obs_B,

                reward_A,

                reward_B,

                done,

                info

            ) = env.step(

                1.0,

                1.0

            )


            if env.step_count % 10 == 0:

                location_A = (

                    env.vehicle_A
                    .get_location()

                )

                location_B = (

                    env.vehicle_B
                    .get_location()

                )


                print()

                print(
                    "--------------------------------------"
                )

                print(
                    f"STEP {env.step_count}"
                )


                print(

                    "Car A | "

                    f"Pos=("
                    f"{location_A.x:.2f}, "
                    f"{location_A.y:.2f}) | "

                    f"Speed="
                    f"{info['speed_A']:.2f} m/s | "

                    f"Distance="
                    f"{info['distance_A']:.2f} m | "

                    f"TTC="
                    f"{info['ttc_A']:.2f} s | "

                    f"Reward="
                    f"{reward_A:.3f}"

                )


                print(

                    "Car B | "

                    f"Pos=("
                    f"{location_B.x:.2f}, "
                    f"{location_B.y:.2f}) | "

                    f"Speed="
                    f"{info['speed_B']:.2f} m/s | "

                    f"Distance="
                    f"{info['distance_B']:.2f} m | "

                    f"TTC="
                    f"{info['ttc_B']:.2f} s | "

                    f"Reward="
                    f"{reward_B:.3f}"

                )


                print(

                    "Risky interaction:",
                    info["risky_interaction"]

                )


                print(

                    "Risk penalty:",
                    f"{info['risk_penalty']:.2f}"

                )


                print(

                    "TTC difference:",
                    f"{info['ttc_difference']:.2f}"

                )


                print(

                    "Passed A:",
                    info["passed_A"],

                    "| Passed B:",
                    info["passed_B"]

                )


            if done:

                print()

                print(
                    "========== EPISODE DONE =========="
                )


                if (

                    info["collision_A"]
                    or
                    info["collision_B"]

                ):

                    print(
                        "Reason: collision"
                    )


                elif info["both_passed"]:

                    print(
                        "Reason: both vehicles passed"
                    )


                elif info["time_limit"]:

                    print(
                        "Reason: maximum steps reached"
                    )


                break


    except KeyboardInterrupt:

        print()

        print(
            "Test interrupted by user."
        )


    finally:

        env.close()



