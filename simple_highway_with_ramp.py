import gym
from gym.spaces import Discrete, Box
import numpy as np
from ray.rllib.env.env_context import EnvContext

class SimpleHighwayRamp(gym.Env):  #Based on OpenAI gym 0.26.1 API

    """This is a 2-lane highway with a single on-ramp on the right side.  It is called "simple" because it does not use
        lanelets, but has a fixed (understood) geometry that is not flexible nor extensible.  It is only
        intended for an initial proof of concept demonstration.

        The intent is that the agent will begin on the on-ramp, driving toward the mainline road and
        attempt to change lanes into the right lane of the highway.  While it is doing that, the environment provides
        zero or more automated vehicles (not learning agents) driving in the right lane of the highway, thus presenting
        a possible safety conflict for the merging agent to resolve by adjusting its speed and merge timing.
            Version 0 will have no mainline vehicles (intended for debugging the geometry & physics)

        This simple enviroment assumes perfect traction and steering response, so that the physics of driving can
        essentially be ignored.  However, there are upper and lower limits on longitudinal and lateral acceleration
        capabilities of each vehicle.

        We will assume that all cars follow their chosen lane's centerlines at all times, except when
        changing lanes.  So there is no steering action required, per se, but there may be opportunity to change lanes
        left or right at certain times.  To support this simplification, the ramp will not join the mainline lane
        directly (e.g. distance between centerlines gradually decreases to 0).  Rather, it will approach asymptotically,
        then run parallel to the mainline lane for some length, then suddenly terminate.  It is in this parallel section
        that a car is able to change from the ramp lane to the mainline lane.  If it does not, it will be considered to
        have crashed (run off the end of the ramp pavement).  The geometry looks something like below, with lanes 0
        and 1 continuing indefinitely, while lane 2 (the on ramp) has a finite length.  Vehicle motion is generally
        from left to right.

            Lane 0  ------------------------------------------------------------------------------------------>
            Lane 1  ------------------------------------------------------------------------------------------>
                                                     ----------------------/
                                                    /
                                                   /
                                                  /
                    Lane 2  ----------------------

        The environment is a continuous flat planar space, but we don't need (x, y) coordinates.  Rather, vehicle locations
        will be represented by their current lane index and distance from the beginning of that lane.

        In this version there is no communication among the vehicles, only (perfect) observations of their own onboard
        sensors.

        In this version the agent magically knows everything going on in the environment, including some connectivities of
        the lane geometry.  Its observation space is (all float32):
            ego_lane_id         - index of the lane the agent is occupying
            ego_x               - agent's distance downtrack in that lane (center of bounding box), m
            ego_speed           - agent's forward speed, m/s
            ego_lane_rem        - distance remaining in the agent's current lane, m
            ego_accel_cmd_cur   - agent's most recent accel_cmd, m/s^2
            ego_accel_cmd_prev1 - agent's next most recent accel_cmd (1 time step old), m/s^2
            ego_accel_cmd_prev2 - agent's next most recent accel_cmd (2 time steps old), m/s^2
            ego_lane_cmd        - agent's most recent lane_change_cmd
            ego_lane_cmd_prev1  - agent's next most recent lane_change_cmd (1 time step old)
            ego_lane_cmd_prev2  - agent's next most recent lane_change_cmd (2 time steps old)
            adj_ln_left_id      - index of the lane that is/will be adjacent to the left of ego lane (-1 if none)
            adj_ln_left_conn_a  - dist from agent to where adjacent lane first joins ego lane, m
            adj_ln_left_conn_b  - dist from agent to where adjacent lane separates from ego lane, m
            adj_ln_left_rem     - dist from agent to end of adjacent lane, m
            adj_ln_right_id     - index of the lane that is/will be adjacent to the right of ego lane (-1 if none)
            adj_ln_right_conn_a - dist from agent to where adjacent lane first joins ego lane, m
            adj_ln_right_conn_b - dist from agent to where adjacent lane separates from ego lane, m
            adj_ln_right_rem    - dist from agent to end of adjacent lane, m
            n1_lane_id          - neighbor vehicle 1, index of the lane occupied by that vehicle
            n1_x                - neighbor vehicle 1, vehicle's dist downtrack in its current lane (center of bounding box), m
            n1_speed            - neighbor vehicle 1, vehicle's forward speed, m/s
            n1_lane_rem         - neighbor vehicle 1, distance remaining in that vehicle's current lane, m
            n2_lane_id          - neighbor vehicle 1, index of the lane occupied by that vehicle
            n2_x                - neighbor vehicle 1, vehicle's dist downtrack in its current lane (center of bounding box), m
            n2_speed            - neighbor vehicle 1, vehicle's forward speed, m/s
            n2_lane_rem         - neighbor vehicle 1, distance remaining in that vehicle's current lane, m
            n3_lane_id          - neighbor vehicle 1, index of the lane occupied by that vehicle
            n3_x                - neighbor vehicle 1, vehicle's dist downtrack in its current lane (center of bounding box), m
            n3_speed            - neighbor vehicle 1, vehicle's forward speed, m/s
            n3_lane_rem         - neighbor vehicle 1, distance remaining in that vehicle's current lane, m
            Note:  lane IDs are always non-negative; if adj_ln_*_id is -1 then the other respective values on that side
                   are meaningless, as there is no lane.

        The agent's action space is continuous and contains the following possible actions:
            accel_cmd           - the desired forward acceleration, m/s^2
            lane_change_cmd     - a pseudo-discrete value with the following meanings:
                                    val < -0.5: change lanes left
                                    -0.5 < val < 0.5: stay in lane
                                    0.5 < val: change lanes right

        Lane connectivities are defined by three parameters, as shown in the following series of diagrams.  These
        parameters work the same whether they describe a lane on the right or left of the ego vehicle's lane, so we only
        show the case of an adjacent lane on the right side.  In this diagram, 'x' is the agent vehicle location.

        Case 1, on-ramp:
                           |<...............rem....................>|
                           |<................B.....................>|
                           |<....A.....>|                           |
                           |            |                           |
            Ego lane  ----[x]---------------------------------------------------------------------------->
                                        /---------------------------/
            Adj lane  -----------------/

        ==============================================================================================================

        Case 2, exit ramp:
                           |<..................................rem.......................................>
                           |<.........B..........>|
                           | A = 0                |
                           |                      |
            Ego lane  ----[x]---------------------------------------------------------------------------->
            Adj lane  ----------------------------\
                                                  \------------------------------------------------------>

        ==============================================================================================================

        Case 3, mainline lane drop:
                           |<...............rem....................>|
                           |<................B.....................>|
                           | A = 0                                  |
                           |                                        |
            Ego lane  ----[x]---------------------------------------------------------------------------->
            Adj lane  ----------------------------------------------/

        Simulation notes:
        + The simulation is run at a configurable time step size.
        + All roadways have the same posted speed limit.
        + Cars are only allowed to go forward, but not above the posted speed limits, so speed values are restricted to the
          range [0, speed_limit].
        + The desired accelerations may not achieve equivalent actual accelerations, due to inertia and traction constraints.
        + If a lane change is desired it will take multiple time steps to complete.
        + Vehicles are modeled as simple rectangular boxes.  Each vehicle's width fits within one lane, but when it is in a
          lane change state, it is considered to fully occupy both the lane it is departing and the lane it is moving toward.
        + If two (or more) vehicles's bounding boxes touch or overlap, they will be considered to have crashed, which ends
          the episode.
        + If any vehicle drives off the end of a lane, it is a crash ane ends the episode.
        + If a lane change is requested where no target lane exists, it is considered a crash and ends the episode.
        + If there is no crash, but all vehicles exit the indefinite end of a lane, then the episode is complete.

        Agent rewards are provided by a separate reward function.  The reward logic is documented there.
    """

    metadata = {"render_modes": None}
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    TIME_STEP = 0.2 #seconds
    OBS_SIZE = 30

    def __init__(self, config: EnvContext, seed=None, render_mode=None):
        """Initialize an object of this class."""

        #print("\n///// SimpleHighwayRamp init: config = ", config)

        # Store the arguments
        #self.seed = seed #Ray 2.0.0 chokes on the seed() method if this is defined (it checks for this attribute also)
        self.render_mode = render_mode

        # Define the essential attributes required of any Env object
        lower_obs = np.array([...]) #TODO
        upper_obs = np.array([...])

        self.observation_space = Box(low=lower_obs, high=upper_obs, shape=(self.OBS_SIZE), dtype=np.float32)
        self.action_space = Box(low=-2.0, high=2.0, shape=(2), dtype=np.float32)







        #assert render_mode is None or render_mode in self.metadata["render_modes"]
        #self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def seed(self, seed=None):
        """A required method that is apparently not yet supported in Ray 2.0.0."""
        pass
        #print("///// In environment seed - incoming seed value = ", seed)
        #self.seed = seed
        #super().seed(seed)


    def reset(self, seed=None, options=None):
        """Reinitializes the environment to prepare for a new episode.  This must be called before
            making any calls to step().
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        #self.seed = seed #okay to pass it to the parent class, but don't store a local member copy!

        # options may be a dict that can specify additional configurations - unique to each particular env
        if options is not None:
            raise ValueError("reset() called with options, but options are not used in this environment.")

        self.cur_pos = 0
        return [self.cur_pos]


    def step(self, action):
        """Executes a single time step of the environment.  Determines how the input actions will alter the
            simulated world and returns the resulting observations to the agent.
        """
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos

        # According to gym docs, return tuple should have 5 elements:
        #   obs
        #   reward
        #   done
        #   truncated (bool) - this one appears to not be supported in gym 0.26.1
        #   info
        return [self.cur_pos], 1.0 if done else -0.02, done, {}


    def close(self):
        """Closes any resources that may have been used during the simulation."""
        pass
