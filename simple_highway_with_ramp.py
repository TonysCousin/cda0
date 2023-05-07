from collections import deque
from statistics import mean
from typing import Tuple, Dict
import math
import time
import gymnasium
from gymnasium.spaces import Box
import numpy as np
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.tune.logger import pretty_print
from perturbation_control import PerturbationController

perturb_ctrl = PerturbationController() #create this outside the class and curriculum function so both can access it


class LaneChange:
    CHANGE_LEFT     = -1
    STAY_IN_LANE    = 0
    CHANGE_RIGHT    = 1


class SimpleHighwayRamp(TaskSettableEnv):  #Based on OpenAI gym 0.26.1 API

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

        The environment is a continuous flat planar space, with the X-axis origin at the left end of lanes 0 & 1. The
        location of any vehicle is represented by its X value and lane ID (which constrains Y), so the Y origin is
        arbitrary, and only used for graphical output.

        In this version there is no communication among the vehicles, only (perfect) observations of their own onboard
        sensors.

        Observation space:  In this version the agent magically knows everything going on in the environment, including
        some connectivities of the lane geometry.  Its observation space is described in the __init__() method (all floats).

        Action space:  continuous, with the following elements (real world values, unscaled):
            target_speed        - the desired forward speed, m/s. Values are in [0, MAX_SPEED].
            target_lane         - indicator of the ID of the currently desired lane, offset by 1. Since this is a float,
                                    and needs to be kept in [-1, 1], we will interpret it loosely as the lane ID - 1,
                                    which can then represent 3 possible choices.  Specifically,
                                    [-1, -0.5)      = lane 0
                                    [-0.5, +0.5]    = lane 1
                                    (0.5, 1]        = lane 2
                                    If a lane is chosen that doesn't exist at the current X location, it will be treated
                                    as an illegal lane change maneuver.

        Lane connectivities are defined by three parameters that define the adjacent lane, as shown in the following
        series of diagrams.  These parameters work the same whether they describe a lane on the right or left of the
        ego vehicle's lane, so we only show the case of an adjacent lane on the right side.  In this diagram, '[x]'
        is the agent (ego) vehicle location.

        Case 1, adjacent to on-ramp:
                           |<...............rem....................>|
                           |<................B.....................>|
                           |<....A.....>|                           |
                           |            |                           |
            Ego lane  ----[x]---------------------------------------------------------------------------->
                                        /---------------------------/
            Adj lane  -----------------/

        ==============================================================================================================

        Case 2, adjacent to exit ramp:
                           |<..................................rem.......................................>
                           |<.........B..........>|
                           | A < 0                |
                           |                      |
            Ego lane  ----[x]---------------------------------------------------------------------------->
            Adj lane  ----------------------------\
                                                  \------------------------------------------------------>

        ==============================================================================================================

        Case 3, adjacent to mainline lane drop:
                           |<...............rem....................>|
                           |<................B.....................>|
                           | A < 0                                  |
                           |                                        |
            Ego lane  ----[x]---------------------------------------------------------------------------->
            Adj lane  ----------------------------------------------/

        Case 4, two parallel lanes indefinitely long:  no diagram needed, but A < 0 and B = inf, rem = inf.


        Simulation notes:
        + The simulation is run at a configurable time step size.
        + All lanes have the same posted speed limit.
        + Cars are only allowed to go forward; they may exceed the posted speed limits, but have a max physical  limit,
          so speeds are in the range [0, max_speed].
        + The desired accelerations may not achieve equivalent actual accelerations, due to inertia and traction constraints.
        + If a lane change is commanded it will take multiple time steps to complete.
        + Vehicles are modeled as simple rectangular boxes.  Each vehicle's width fits within one lane, but when it is in a
          lane change state, it is considered to fully occupy both the lane it is departing and the lane it is moving toward.
        + If two (or more) vehicles' bounding boxes touch or overlap, they will be considered to have crashed, which ends
          the episode.
        + If any vehicle drives off the end of a lane, it is driving off-road and ends the episode.
        + If a lane change is requested where no target lane exists, it is driving off-road and ends the episode.
        + If there is no crash or off-road, but the ego vehicle exits the indefinite end of a lane, then the episode ends
          successfully.
        + The environment supports curriculum learning with multiple levels of difficulty. The levels are:
            0 = solo agent drives a (possibly short) straight lane to the end without departing the roadway with limited init spd;
                focus is on making legal lane changes in small numbers (lanes 0 & 1 only)
            1 = level 0 but always start near beginning of track with full range of possible initial speeds
            2 = level 1 plus added emphasis on minimizing jerk and speed changes
            3 = level 2 plus solo agent driving on entry ramp (lane 2), and forced to change lanes to complete the course
            4 = level 3 plus 3 sequential, constant-speed vehicles in lane 1
            5 = level 3 plus 3 randomly located, constant-speed vehicles anywhere on the track
            6 = level 3 plus 3 randomly located, variable-speed vehicles anywhere on the track

        Agent rewards are provided by a separate reward function.  The reward logic is documented there.
    """

    metadata = {"render_modes": None}
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    OBS_SIZE                = 53
    VEHICLE_LENGTH          = 20.0      #m
    NUM_LANES               = 3         #total number of unique lanes in the scenario
    MAX_SPEED               = 36.0      #vehicle's max achievable speed, m/s
    MAX_ACCEL               = 3.0       #vehicle's max achievable acceleration (fwd or backward), m/s^2
    MAX_JERK                = 4.0       #max desirable jerk for occupant comfort, m/s^3
    ROAD_SPEED_LIMIT        = 29.1      #Roadway's legal speed limit on all lanes, m/s (29.1 m/s = 65 mph)
    SCENARIO_LENGTH         = 2000.0    #total length of the roadway, m
    SCENARIO_BUFFER_LENGTH  = 200.0     #length of buffer added to the end of continuing lanes, m
    NUM_NEIGHBORS           = 3         #total number of neighbor vehicles in scenario (may not be moving)
    OBS_ZONE_LENGTH         = 2.0 * ROAD_SPEED_LIMIT #the length of a roadway observation zone, m
    #TODO: make this dependent upon time step size:
    HALF_LANE_CHANGE_STEPS  = 3.0 / 0.5 // 2    #num steps to get half way across the lane (equally straddling both)
    TOTAL_LANE_CHANGE_STEPS = 2 * HALF_LANE_CHANGE_STEPS
    MAX_STEPS_SINCE_LC      = 60        #largest num time steps we will track since previous lane change
    NUM_DIFFICULTY_LEVELS   = 5         #num levels of environment difficulty for the agent to learn; see descrip above

    def __init__(self,
                 config:        EnvContext,             #dict of config params
                 seed:          int             = None, #seed for PRNG
                 render_mode:   int             = None  #Ray rendering info, unused in this version
                ):

        """Initialize an object of this class.  Config options are:
            time_step_size: duration of a time step, s (default = 0.5)
            debug:          level of debug printing (0=none (default), 1=moderate, 2=full details)
            training:       (bool) True if this is a training run, else it is inference (affects initial conditions)
            init_ego_lane:  lane ID that the agent vehicle begins in (default = 2)
            init_ego_speed: initial speed of the agent vehicle, m/s (defaults to random in [5, 20])
            init_ego_dist:  initial downtrack location of the agent vehicle from lane begin, m (defaults to random in [0, 200])
            randomize_start_dist: (bool) True if ego vehicle start location is to be randomized; only applies to level 0
            neighbor_speed: initial speed of the neighbor vehicles, m/s (default = 29.1)
            neighbor_start_loc: initial location of neighbor vehicle N3 (the rear-most vehicle), m (default = 0)
            stopper:        the object used to render experiment stopping decisions (default = None)
            difficulty_level: the fixed level of difficulty of the environment in [0, NUM_DIFFICULTY_LEVELS) (default = 0)
            burn_in_iters:  num iterations that must pass before a difficulty level promotion is possible
       """

        super().__init__()

        # Store the arguments
        #self.seed = seed #Ray 2.0.0 chokes on the seed() method if this is defined (it checks for this attribute also)
        #TODO: try calling self.seed() without storing it as an instance attribute
        self.prng = np.random.default_rng(seed = seed)
        self.render_mode = render_mode

        self._set_initial_conditions(config)
        if self.debug > 0:
            print("\n///// SimpleHighwayRamp init: config = ", config)

        # Define the vehicles used in this scenario - the ego vehicle (where the agent lives) is index 0
        self.vehicles = []
        for i in range(SimpleHighwayRamp.NUM_NEIGHBORS + 1):
            v = Vehicle(self.time_step_size, SimpleHighwayRamp.MAX_JERK, self.debug)
            self.vehicles.append(v)

        #
        #..........Define the observation space
        #
        # A key portion of the obs space is a representation of spatial zones around the ego vehicle. These zones
        # move with the vehicle, and are a schematic representation of the nearby roadway situation. That is, they
        # don't include any lane shape geometry, representing every lane as straight, and physically adjacent to its
        # next lane. This is possible because we ASSUME that the vehicles are operating well inside their performance
        # limits, so that road geometry won't affect their ability to change lanes, accelerate or decelerate at any
        # desired rate. The zones are arranged as in the following diagram.
        #
        #              +----------+----------+----------+
        #              |  zone 7  |  zone 4  |  zone 1  |
        #   +----------+----------+----------+----------+
        #   |  zone 9  |  ego veh |  zone 5  |  zone 2  | >>>>> direction of travel
        #   +----------+----------+----------+----------+
        #              |  zone 8  |  zone 6  |  zone 3  |
        #              +----------+----------+----------+
        #
        # All zones are the same size, with their width equal to the lane width. Zone length is nominally 2 sec of
        # travel distance (at the posted speed limit). The ego vehicle is always centered in its zone, longitudinally.
        # If any vehicle is in the process of changing lanes, it will be observed to be in both adjacent zones through
        # the full lane change process. However, since the grid is centered on the ego vehicle, that vehicle will be
        # handled specially when it comes to lane change maneuvers, looking into either zone 7 or zone 8 concerning a
        # possible crash.
        #
        # Each zone will provide the following set of information:
        #   Is it drivable? E.g. is there a lane in that relative position
        #   Is the zone reachable from ego lane? I.e. lateral motion is possible and legal along the full length of the zone.
        #   Is there a neighbor vehicle in the zone? No more than one vehicle will be allowed in any given zone.
        #   Occupant's X location within the zone, if occupant exists ((X - Xrear) / zone length)
        #   Occupant's speed relative to ego vehicle, if occupant exists (delta-S / speed limit)

        # Indices into the observation vector
        self.EGO_LANE_ID        =  0 #index of the lane the agent is occupying
        self.EGO_X              =  1 #agent's X coordinate (center of bounding box), m
        self.EGO_SPEED          =  2 #agent's forward speed, m/s
        self.EGO_SPEED_PREV     =  3 #agent's actual speed in previous time step, m/s
        self.EGO_LANE_REM       =  4 #distance remaining in the agent's current lane, m     #TODO: is this redundant with zone info?
        self.EGO_DESIRED_LN     =  5 #agent's most recent desired lane action (from prev time step)
        self.STEPS_SINCE_LN_CHG =  6 #num time steps since the previous lane change was initiated
        self.NEIGHBOR_IN_EGO_ZONE = 7 #if any neighbor vehicle is in the ego zone, then this value will be 1 if it is in front of the
                                     # ego vehicle or -1 if it is behind the ego vehicle. If no neigbhor is present, value = 0

        self.Z1_DRIVEABLE       =  8 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z1_REACHABLE       =  9 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z1_OCCUPIED        = 10 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z1_ZONE_X          = 11 #occupant's X location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z1_REL_SPEED       = 12 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z2_DRIVEABLE       = 13 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z2_REACHABLE       = 14 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z2_OCCUPIED        = 15 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z2_ZONE_X          = 16 #occupant's X location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z2_REL_SPEED       = 17 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z3_DRIVEABLE       = 18 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z3_REACHABLE       = 19 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z3_OCCUPIED        = 20 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z3_ZONE_X          = 21 #occupant's X location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z3_REL_SPEED       = 22 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z4_DRIVEABLE       = 23 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z4_REACHABLE       = 24 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z4_OCCUPIED        = 25 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z4_ZONE_X          = 26 #occupant's X location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z4_REL_SPEED       = 27 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z5_DRIVEABLE       = 28 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z5_REACHABLE       = 29 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z5_OCCUPIED        = 30 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z5_ZONE_X          = 31 #occupant's X location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z5_REL_SPEED       = 32 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z6_DRIVEABLE       = 33 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z6_REACHABLE       = 34 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z6_OCCUPIED        = 35 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z6_ZONE_X          = 36 #occupant's X location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z6_REL_SPEED       = 37 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z7_DRIVEABLE       = 38 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z7_REACHABLE       = 39 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z7_OCCUPIED        = 40 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z7_ZONE_X          = 41 #occupant's X location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z7_REL_SPEED       = 42 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z8_DRIVEABLE       = 43 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z8_REACHABLE       = 44 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z8_OCCUPIED        = 45 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z8_ZONE_X          = 46 #occupant's X location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z8_REL_SPEED       = 47 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z9_DRIVEABLE       = 48 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z9_REACHABLE       = 49 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z9_OCCUPIED        = 50 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z9_ZONE_X          = 51 #occupant's X location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z9_REL_SPEED       = 52 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        max_rel_speed = SimpleHighwayRamp.MAX_SPEED / SimpleHighwayRamp.ROAD_SPEED_LIMIT
        lower_obs = np.zeros((SimpleHighwayRamp.OBS_SIZE)) #most values are 0, so only the others are explicitly set below
        lower_obs(self.NEIGHBOR_IN_EGO_ZONE)= -1.0
        lower_obs[self.Z1_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z2_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z3_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z4_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z5_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z6_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z7_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z8_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z9_REL_SPEED]        = -max_rel_speed

        upper_obs = np.ones(SimpleHighwayRamp.OBS_SIZE) #most values are 1
        upper_obs[self.EGO_LANE_ID]         = SimpleHighwayRamp.NUM_LANES - 1
        upper_obs[self.EGO_X]               = SimpleHighwayRamp.SCENARIO_LENGTH
        upper_obs[self.EGO_SPEED]           = SimpleHighwayRamp.MAX_SPEED
        upper_obs[self.EGO_SPEED_PREV]      = SimpleHighwayRamp.MAX_SPEED
        upper_obs[self.EGO_LANE_REM]        = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        upper_obs[self.EGO_DESIRED_LN]      = SimpleHighwayRamp.NUM_LANES - 1
        upper_obs[self.STEPS_SINCE_LN_CHG]  = SimpleHighwayRamp.MAX_STEPS_SINCE_LC
        upper_obs[self.Z1_REL_SPEED]        = max_rel_speed
        upper_obs[self.Z2_REL_SPEED]        = max_rel_speed
        upper_obs[self.Z3_REL_SPEED]        = max_rel_speed
        upper_obs[self.Z4_REL_SPEED]        = max_rel_speed
        upper_obs[self.Z5_REL_SPEED]        = max_rel_speed
        upper_obs[self.Z6_REL_SPEED]        = max_rel_speed
        upper_obs[self.Z7_REL_SPEED]        = max_rel_speed
        upper_obs[self.Z8_REL_SPEED]        = max_rel_speed
        upper_obs[self.Z9_REL_SPEED]        = max_rel_speed

        self.observation_space = Box(low=lower_obs, high=upper_obs, dtype=np.float)
        if self.debug == 2:
            print("///// observation_space = ", self.observation_space)

        self.obs = np.zeros(SimpleHighwayRamp.OBS_SIZE) #will be returned from reset() and step()
        self._verify_obs_limits("init after space defined")

        #
        #..........Define the action space
        #

        lower_action = np.array([0.0, -1.0])
        upper_action = np.array([ SimpleHighwayRamp.MAX_SPEED,  1.0])
        self.action_space = Box(low=lower_action, high=upper_action, dtype=np.float)
        if self.debug == 2:
            print("///// action_space = ", self.action_space)

        #
        #..........Remaining initializations
        #

        # Create the roadway geometry
        self.roadway = Roadway(self.debug)

        # Other persistent data
        self.lane_change_underway = "none" #possible values: "left", "right", "none"
        self.lane_change_count = 0  #num consecutive time steps since a lane change was begun
        self.total_steps = 0        #num time steps for this trial (worker), across all episodes; NOTE that this is different from the
                                    # total steps reported by Ray tune, which is accumulated over all rollout workers
        self.steps_since_reset = 0  #length of the current episode in time steps
        self.stopped_count = 0      #num consecutive time steps in an episode where vehicle speed is zero
        self.reward_for_completion = True #should we award the episode completion bonus?
        self.episode_count = 0      #number of training episodes (number of calls to reset())
        self.num_crashes = 0        #num crashes with a neighbor vehicle since reset
        self.neighbor_print_latch = True #should the neighbor vehicle info be printed when initiated?
        self.rollout_id = hex(int(self.prng.random() * 65536))[2:].zfill(4) #random int to ID this env object in debug logging
        print("///// Initializing env environment ID {} at level {}".format(self.rollout_id, self.difficulty_level))

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
        if self.debug == 2:
            print("///// init complete.")


    def seed(self, seed=None):
        """A required method that is apparently not yet supported in Ray 2.0.0."""
        pass
        #print("///// In environment seed - incoming seed value = ", seed)
        #self.seed = seed
        #super().seed(seed)


    def set_task(self,
                 task:      int         #ID of the task (lesson difficulty) to simulate, in [0, n]
                ) -> None:
        """Defines the difficulty level of the environment, which can be used for curriculum learning."""

        self.difficulty_level = min(max(task, 0), self.NUM_DIFFICULTY_LEVELS)
        print("\n\n///// Environment difficulty for rollout ID {} set to {}\n".format(self.rollout_id, self.difficulty_level))


    def get_task(self) -> int:
        """Returns the environment difficulty level currently in use."""

        return self.difficulty_level


    def reset(self, *,
              seed:         int             = None,    #reserved for a future version of gym.Environment
              options:      dict            = None
             ) -> Tuple[np.array, dict]:

        """Reinitializes the environment to prepare for a new episode.  This must be called before
            making any calls to step().

            Return tuple includes an array of observation values, plus a dict of additional key-value
            info pairs.

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!
        """

        if self.debug > 0:
            print("\n///// Entering reset")

        # We need the following line to seed self.np_random
        #super().reset(seed=seed) #apparently gym 0.26.1 doesn't implement this method in base class!
        #self.seed = seed #okay to pass it to the parent class, but don't store a local member copy!

        # options may be a dict that can specify additional configurations - unique to each particular env
        if options is not None and len(options) > 0:
            print("\n///// SimpleHighwayRamp.reset: incoming options is: ", options)
            raise ValueError("reset() called with options, but options are not used in this environment.")

        # If the agent is being trained then
        ego_lane_id = None
        ego_x = None
        ego_speed = None
        max_distance = 1.0
        if self.training:
            ego_lane_id = self._select_init_lane() #covers all difficulty levels
            ego_lane_start = self.roadway.get_lane_start_x(ego_lane_id)

            # This area of code is a sandbox for figuring out best approaches to working with curriculum learning and PBT simultaneously.
            # Some tests & comments may seem obtuse as a result.

            # If we are at difficulty level 0-3, then choose widely randomized initial conditions
            if self.difficulty_level < 4:
                ego_x = 0.0
                if self.randomize_start_dist  and  not perturb_ctrl.has_perturb_begun():
                    physical_limit = min(self.roadway.get_total_lane_length(ego_lane_id), SimpleHighwayRamp.SCENARIO_LENGTH) - ego_lane_start - 10.0
                    initial_steps = 60000 #num steps to wait before starting to shrink the max distance
                    if self.total_steps <= initial_steps:
                        max_distance = physical_limit
                    else:
                        max_distance = max((self.total_steps - initial_steps) * (10.0 - physical_limit)/(1.6e6 - initial_steps) + physical_limit,
                                        10.0) #decreases over time steps
                    ego_x = self.prng.random() * max_distance + ego_lane_start
                ego_speed = self.prng.random() * 10.0 + 20.0 #value in [10, 30] m/s

            elif self.difficulty_level < 4: #levels 1, 2, 3
                ego_x = self.prng.random() * 500.0
                ego_speed = self.prng.random() * SimpleHighwayRamp.MAX_SPEED #any physically possible value

            elif self.difficulty_level == 4:
                # We want to sync the agent in lane 2 to force it to avoid a collision by positioning it right next to the neighbors
                if ego_lane_id == 2:
                    ego_x = self.prng.random() * 3.0*SimpleHighwayRamp.VEHICLE_LENGTH
                    loc = int(self.prng.random()*5.0)
                    ego_speed = self.initialize_ramp_vehicle_speed(loc, ego_x)
                else:
                    ego_x = self.prng.random() * 500.0
                    ego_speed = self.prng.random() * (SimpleHighwayRamp.MAX_SPEED - 15.0) + 15.0 #not practical to train at really low speeds

                # Code block inspired from above to support training level 4 from scratch - it needs to learn how to find the finish line
                initial_steps = 60000 #num steps to wait before starting to shrink the max distance
                if self.randomize_start_dist  and  self.total_steps <= initial_steps:
                    physical_limit = min(self.roadway.get_total_lane_length(ego_lane_id), SimpleHighwayRamp.SCENARIO_LENGTH) - ego_lane_start - 10.0
                    ego_x = self.prng.random() * physical_limit

            else: #levels 5 and up
                ego_x = self.prng.random() * 500.0
                ego_speed = self.prng.random() * (SimpleHighwayRamp.MAX_SPEED - 15.0) + 15.0 #not practical to train at really low speeds

        # Else, we are doing inference, so allow coonfigrable overrides if present
        else:
            ego_lane_id = int(self.prng.random()*3) if self.init_ego_lane is None  else  self.init_ego_lane
            ego_x = self.prng.random() * 3.0*SimpleHighwayRamp.VEHICLE_LENGTH if self.init_ego_dist is None  else  self.init_ego_dist + ego_lane_start

            # If difficulty level 4, then always use the config value for merge relative position
            if self.difficulty_level == 4  and  ego_lane_id == 2:
                ego_speed = self.initialize_ramp_vehicle_speed(self.merge_relative_pos, ego_x)
            else:
                ego_speed = self.prng.random() * 31.0 + 4.0 if self.init_ego_speed is None  else  self.init_ego_speed

        # If this difficulty level uses neighbor vehicles then initialize their speed and starting point
        n_loc = 0.0
        n_speed = 0.0
        if self.difficulty_level == 4: #steady speed vehicles in lane 1
            n_loc = max(self.neighbor_start_loc + self.prng.random()*6.0*SimpleHighwayRamp.VEHICLE_LENGTH, 0.0)
            n_speed = self.neighbor_speed * (self.prng.random()*0.2 + 0.9) #assigned speed +/- 10%
            if self.prng.random() > 0.9: #sometimes there will be no neighbor vehicles participating
                n_speed = 0.0
            if self.neighbor_print_latch:
                print("///// reset worker {}: Neighbor vehicles on the move in level 4. Episode {}, step {}, loc = {:.1f}, speed = {:.1f}"
                        .format(self.rollout_id, self.episode_count, self.total_steps, n_loc, n_speed))
                self.neighbor_print_latch = False
        elif self.difficulty_level > 4:
            raise NotImplementedError("///// Neighbor vehicle motion not defined for difficulty level {}".format(self.difficulty_level))

        # Neighbor vehicles always go a constant speed, always travel in lane 1
        #TODO: revise this for levels 4+
        n_lane_id = 1
        self.vehicles[1].lane_id = n_lane_id
        self.vehicles[1].x = n_loc + 6.0*SimpleHighwayRamp.VEHICLE_LENGTH #in front of vehicle n2
        self.vehicles[1].speed = n_speed
        self.vehicles[1].lane_change_status = "none"
        self.vehicles[2].lane_id = n_lane_id
        self.vehicles[2].x = n_loc + 3.0*SimpleHighwayRamp.VEHICLE_LENGTH #in front of vehicle n3
        self.vehicles[2].speed = n_speed
        self.vehicles[2].lane_change_status = "none"
        self.vehicles[3].lane_id = n_lane_id
        self.vehicles[3].x = n_loc + 0.0 #at end of the line of 3 neighbors
        self.vehicles[3].speed = n_speed
        self.vehicles[3].lane_change_status = "none"
        if self.debug > 1:
            print("      in reset: vehicles = ")
            for i, v in enumerate(self.vehicles):
                v.print(i)
        if SimpleHighwayRamp.NUM_NEIGHBORS != 3:
            raise NotImplementedError("///// Incorrect number of neighbor vehicles initialized!")

        # If the ego vehicle is in lane 1 (where the neighbor vehicles are), then we need to initialize its position so that it isn't
        # going to immediately crash with them (n1 is always the farthest downtrack and n3 is always the farthest uptrack). Give it
        # more room if going in front of the neighbors, as ego has limited accel and may be starting much slower than they are.
        #TODO: when difficulty levels > 3 are implemented, this needs to account for vehicles in other lanes also.
        if ego_lane_id == 1:
            min_loc = max(self.vehicles[3].x - 4.0*SimpleHighwayRamp.VEHICLE_LENGTH, 0.0)
            max_loc = self.vehicles[1].x + 10.0*SimpleHighwayRamp.VEHICLE_LENGTH
            midway = 0.5*(max_loc - min_loc) + min_loc
            if midway < ego_x < max_loc:
                ego_x = max_loc
                if self.debug > 0:
                    print("///// reset initializing agent to: lane = {}, speed = {:.2f}, x = {:.2f}".format(ego_lane_id, ego_speed, ego_x))
            elif min_loc < ego_x <= midway:
                ego_x = min_loc
                if self.debug > 0:
                    print("///// reset initializing agent to: lane = {}, speed = {:.2f}, x = {:.2f}".format(ego_lane_id, ego_speed, ego_x))
        #print("///// reset: training = {}, ego_lane_id = {}, ego_x = {:.2f}, ego_speed = {:.2f}".format(self.training, ego_lane_id, ego_x, ego_speed))
        self._verify_obs_limits("reset after initializing local vars")

        # Reinitialize the whole observation vector
        ego_rem, lid, la, lb, l_rem, rid, ra, rb, r_rem = self.roadway.get_current_lane_geom(ego_lane_id, ego_x)
        self.vehicles[0].lane_id = ego_lane_id
        self.vehicles[0].x = ego_x
        self.vehicles[0].speed = ego_speed
        self.vehicles[0].lane_change_status = "none"

        self.obs = np.zeros(SimpleHighwayRamp.OBS_SIZE)
        self.obs[self.EGO_LANE_ID]          = ego_lane_id
        self.obs[self.EGO_X]                = ego_x
        self.obs[self.EGO_SPEED]            = ego_speed
        self.obs[self.EGO_SPEED_PREV]       = ego_speed
        self.obs[self.EGO_LANE_REM]         = ego_rem
        self.obs[self.EGO_DESIRED_LN]       = ego_lane_id #this is feedback from previous timestep, so okay to initialize it like this
        self.obs[self.STEPS_SINCE_LN_CHG]   = SimpleHighwayRamp.MAX_STEPS_SINCE_LC
        self.update_obs_zones()
        self._verify_obs_limits("reset after populating main obs with ego stuff")

    def update_obs_zones(self):
        """Updates the observation vector data for each of the roadway zones, based on ego state and current neighbor vehicle states."""

        # Determine offsets in the obs vector for zone columns and rows
        base = self.Z1_DRIVEABLE
        num_zone_fields = self.Z2_DRIVEABLE - base

        # Clear all zone info from previous time step
        for z in range(9):
            self.obs[base + z + 0] = 0.0 #drivable
            self.obs[base + z + 1] = 0.0 #reachable
            self.obs[base + z + 2] = 0.0 #occupied
            self.obs[base + z + 3] = 0.0 #x
            self.obs[base + z + 4] = 0.0 #speed
        self.obs[self.NEIGHBOR_IN_EGO_ZONE] = 0.0




        #TODO: move this up and handle the first two data elements before looping on neighbors
        # Get the current roadway geometry
        ego_lane_id = self.vehicles[0].lane_id
        ego_rem, lid, la, lb, l_rem, rid, ra, rb, r_rem = self.roadway.get_current_lane_geom(ego_lane_id, ego_x)












        # Loop through the neighbor vehicles
        ego_x = self.vehicles[0].x
        for neighbor_idx in range(1, SimpleHighwayRamp.NUM_NEIGHBORS):
            nv = self.vehicles[neighbor_idx]

            # Find which zone column it is in (relative lane), if any (could be 2 lanes away) (ego is in column 1, lanes are 0-indexed, left-to-right)
            column = nv.lane_id - ego_lane_id + 1

            # Find which zone row it is in, if any (could be too far away)
            row = 0
            dist_ahead_of_ego = nv.x - ego_x
            if dist_ahead_of_ego > 2.5 * SimpleHighwayRamp.OBS_ZONE_LENGTH: #way out front somewhere
                row = 0
            elif dist_ahead_of_ego > 1.5 * SimpleHighwayRamp.OBS_ZONE_LENGTH:
                row = 1
            elif dist_ahead_of_ego > 0.5 * SimpleHighwayRamp.OBS_ZONE_LENGTH:
                row = 2
            elif dist_ahead_of_ego > -0.5 * SimpleHighwayRamp.OBS_ZONE_LENGTH:
                row = 3
            elif dist_ahead_of_ego > -1.5 * SimpleHighwayRamp.OBS_ZONE_LENGTH:
                row = 4
            # Else too far behind to consider, so allow row value of 0 for this case also

            # If the neighbor is too far away, no further consideration needed
            if column < 0  or  column > 2  or  row == 0:
                continue

            # Neighbor is within our obs zone grid - if it is within the ego zone then
            if column == 1  and  row == 3:

                # Set the flag - nothing else needed
                self.obs[self.NEIGHBOR_IN_EGO_ZONE] = 1.0
                if dist_ahead_of_ego < 0.0:
                    self.obs[self.NEIGHBOR_IN_EGO_ZONE] = -1.0

                if self.debug > 1:
                    print("///// update_obs_zones: neighbor {} is in ego zone!".format(neighbor_idx))

            # Else get its offset into the obs vector
            else:
                zone = 0
                if row < 4:
                    zone = 3*(row - 1) + column + 1
                else:
                    assert column == 1, "///// ERROR: update_obs_zones found column = {}, row = {} for neighbor {}".format(column, row, neighbor_idx)
                    zone = 9

                offset = base + (zone - 1)*num_zone_fields

                # Since we've identified a neighbor vehicle in this zone, flag it as occupied
                self.obs[base + offset + 2] = 1.0

                # Set the neighbor's relative location within the zone
                zone_rear_x = ego_x + ((2.0 - row) + 0.5)*SimpleHighwayRamp.OBS_ZONE_LENGTH
                rel_x = (nv.x - zone_rear_x) / SimpleHighwayRamp.OBS_ZONE_LENGTH
                self.obs[base + offset + 3] = rel_x

                # Set the neighbor's relative speed
                self.obs[base + offset + 4] = (nv.speed - self.vehicles[0].speed) / SimpleHighwayRamp.ROAD_SPEED_LIMIT

                if self.debug > 1:
                    print("///// update_obs_zones: neighbor {} has column = {}, row = {}, zone = {}, offset = {}".format(neighbor_idx, column, row, zone, offset))




        self.EGO_LANE_ID        =  0 #index of the lane the agent is occupying
        self.EGO_X              =  1 #agent's X coordinate (center of bounding box), m
        self.EGO_SPEED          =  2 #agent's forward speed, m/s
        self.EGO_SPEED_PREV     =  3 #agent's actual speed in previous time step, m/s
        self.EGO_LANE_REM       =  4 #distance remaining in the agent's current lane, m     #TODO: is this redundant with zone info?
        self.EGO_DESIRED_LN     =  5 #agent's most recent lane_change_cmd
        self.STEPS_SINCE_LN_CHG =  6 #num time steps since the previous lane change was initiated

        self.Z1_DRIVEABLE       =  7 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z1_REACHABLE       =  8 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z1_OCCUPIED        =  9 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z1_ZONE_X          = 10 #occupant's X location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z1_REL_SPEED       = 11 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)







        self._verify_obs_limits("reset after populating neighbor vehicles in obs")

        # Other persistent data
        self.lane_change_underway = "none"
        self.lane_change_count = 0
        self.steps_since_reset = 0
        self.stopped_count = 0
        self.reward_for_completion = True
        self.episode_count += 1
        self.accel_hist.clear()
        self.speed_hist.clear()

        self._verify_obs_limits("end of reset")

        if self.debug > 0:
            print("///// End of reset().")
        return self.obs, {}


    def step(self,
                action  : list      #list of floats; 0 = speed command, 1 = desired lane
            ) -> Tuple[np.array, float, bool, bool, Dict]:
        """Executes a single time step of the environment.  Determines how the input actions will alter the
            simulated world and returns the resulting observations to the agent.

            Return is array of new observations, new reward, done flag, truncated flag, and a dict of additional info.

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!
        """

        if self.debug > 0:
            print("\n///// Entering step(): action = ", action)
            print("      vehicles array contains:")
            for i, v in enumerate(self.vehicles):
                v.print(i)

        self.total_steps += 1
        self.steps_since_reset += 1

        #
        #..........Calculate new state for the ego vehicle and determine if episode is complete
        #

        done = False
        return_info = {"reason": "Unknown"}












        #TODO: rewrite this whole block - can probably just use the vehicles vector instead of obs

        # Move all of the neighbor vehicles downtrack (ASSUMES all vehicles are represented contiguously in the obs vector).
        # This doesn't account for possible lane changes, which are handled seperately in the next section.
        # CAUTION: this block depends on the structure of the obs vector indexes for each of the vehicles. Ego vehicle must be
        #           the first one in the loop.
        new_ego_speed = 0.0 #initialize these here to ensure they are in scope througout the method
        new_ego_x = 0.0
        new_ego_rem = 0.0
        for i, v in enumerate(self.vehicles):
            obs_idx = self.EGO_LANE_ID + 3*i
            lane_id = v.lane_id
            if self.debug > 1:
                print("      Advancing vehicle {} with obs_idx = {}, lane_id = {}".format(i, obs_idx, lane_id))
            new_speed_cmd = self.neighbor_speed #non-ego vehicles are constant speed
            cur_speed = self.neighbor_speed
            prev_speed = self.neighbor_speed
            if i == 0: #this is the ego vehicle
                new_speed_cmd = action[0]
                cur_speed = self.obs[self.EGO_SPEED]
                prev_speed = self.obs[self.EGO_SPEED_PREV]
            new_speed, new_x = v.advance_vehicle_spd(new_speed_cmd, cur_speed, prev_speed)
            if new_x > SimpleHighwayRamp.SCENARIO_LENGTH:
                new_x = SimpleHighwayRamp.SCENARIO_LENGTH #limit it to avoid exceeding NN input validation rules
                if i == 0: #the ego vehicle has crossed the finish line; episode is now complete
                    done = True
                    return_info["reason"] = "Success; end of scenario"
            if self.debug > 1:
                print("      Vehicle {} advanced with new_speed_cmd = {:.2f}. new_speed = {:.2f}, new_x = {:.2f}"
                        .format(i, new_speed_cmd, new_speed, new_x))

            if i == 0: #ego vehicle
                self.obs[obs_idx + 1] = new_x
                self.obs[obs_idx + 2] = new_speed
                new_ego_x = new_x
                new_ego_speed = new_speed
            else:
                self.obs[obs_idx + 1] = new_x - new_ego_x
                self.obs[obs_idx + 2] = new_speed - new_ego_speed
                # Neighbor vehicles don't track distance remaining

        # Update the distance remaining for the ego vehicle in its current lane (this will be slightly negative if vehicle
        # ran off the end of a terminating lane, but that condition is tested further down)
        new_ego_rem, _, _, _, _, _, _, _, _ = self.roadway.get_current_lane_geom(int(self.obs[self.EGO_LANE_ID]), new_ego_x)
        self.obs[self.EGO_LANE_REM] = new_ego_rem
        self._verify_obs_limits("step after moving vehicles forward")

        # Determine if we are beginning or continuing a lane change maneuver.
        # Accept a lane change command that lasts for several time steps or only one time step.  Once the first
        # command is received (when currently not in a lane change), then start the maneuver and ignore future
        # lane change commands until the underway maneuver is complete, which takes several time steps.
        # It's legal, but not desirable, to command opposite lane change directions in consecutive time steps.
        # TODO future: replace instance variables lane_change_underway and lane_id with those in vehicle[0]
        ran_off_road = False
        desired_lane = int(math.floor(action[1] + 0.5)) + 1 #maps [-1, 1] into (0, 1, 2)
        lc_cmd = LaneChange.STAY_IN_LANE
        if desired_lane < self.obs[self.EGO_LANE_ID]: #could be different by 1 or 2; assumes lane IDs increase left-to-right
            lc_cmd = LaneChange.CHANGE_LEFT
        elif desired_lane > self.obs[self.EGO_LANE_ID]:
            lc_cmd = LaneChange.CHANGE_RIGHT

        if lc_cmd != LaneChange.STAY_IN_LANE  or  self.lane_change_underway != "none":
            if self.lane_change_underway == "none": #count should always be 0 in this case, so initiate a new count
                if lc_cmd == LaneChange.CHANGE_LEFT:
                    self.lane_change_underway = "left"
                else:
                    self.lane_change_underway = "right"
                self.lane_change_count = 1
                if self.debug > 0:
                    print("      *** New lane change maneuver initiated. lc_cmd = {}, status = {}"
                            .format(lc_cmd, self.lane_change_underway))
            else: #once a lane change is underway, contiinue until complete, regardless of new commands
                self.lane_change_count += 1

        # Check that an adjoining lane is available in the direction commanded until maneuver is complete
        new_ego_lane = int(self.obs[self.EGO_LANE_ID])
        tgt_lane = new_ego_lane
        if self.lane_change_count > 0:

            # If we are still in the original lane then
            if self.lane_change_count <= SimpleHighwayRamp.HALF_LANE_CHANGE_STEPS:
                # Ensure that there is a lane to change into and get its ID
                tgt_lane = self.roadway.get_target_lane(int(self.obs[self.EGO_LANE_ID]), self.lane_change_underway, new_ego_x)
                if tgt_lane < 0:
                    done = True
                    ran_off_road = True
                    return_info["reason"] = "Ran off road; illegal lane change"
                    if self.debug > 1:
                        print("      DONE!  illegal lane change commanded.")

                # Else, we are still going; if we are exactly half-way then change the current lane ID
                elif self.lane_change_count == SimpleHighwayRamp.HALF_LANE_CHANGE_STEPS:
                    new_ego_lane = tgt_lane

            # Else, we have already crossed the dividing line and are now mostly in the target lane
            else:
                coming_from = "left"
                if self.lane_change_underway == "left":
                    coming_from = "right"
                # Ensure the lane we were coming from is still adjoining (since we still have 2 wheels there)
                prev_lane = self.roadway.get_target_lane(tgt_lane, coming_from, new_ego_x)
                if prev_lane < 0: #the lane we're coming from ended before the lane change maneuver completed
                    done = True
                    ran_off_road = True
                    return_info["reason"] = "Ran off road; lane change initiated too late"
                    if self.debug > 1:
                        print("      DONE!  original lane ended before lane change completed.")

        # Get updated metrics of ego vehicle relative to the new lane geometry
        new_ego_rem, lid, la, lb, l_rem, rid, ra, rb, r_rem = self.roadway.get_current_lane_geom(new_ego_lane, new_ego_x)

        # If remaining lane distance has gone away, then vehicle has run straight off the end of the lane, so episode is done
        if new_ego_rem <= 0.0:
            new_ego_rem = 0.0 #clip it so that obs space isn't violated
            if not done:
                done = True
                ran_off_road = True #don't turn this on if we had a successful episode
                return_info["reason"] = "Ran off end of terminating lane"

        # Update counter for time in between lane changes
        if self.obs[self.STEPS_SINCE_LN_CHG] < SimpleHighwayRamp.MAX_STEPS_SINCE_LC:
            self.obs[self.STEPS_SINCE_LN_CHG] += 1

        # If current lane change is complete, then reset its state and counter
        if self.lane_change_count >= SimpleHighwayRamp.TOTAL_LANE_CHANGE_STEPS:
            self.lane_change_underway = "none"
            self.lane_change_count = 0
            self.obs[self.STEPS_SINCE_LN_CHG] = SimpleHighwayRamp.TOTAL_LANE_CHANGE_STEPS

        self.vehicles[0].lane_id = new_ego_lane
        self.vehicles[0].lane_change_status = self.lane_change_underway
        if self.debug > 0:
            print("      step: done lane change. underway = {}, new_ego_lane = {}, tgt_lane = {}, count = {}, done = {}, steps since = {}"
                    .format(self.lane_change_underway, new_ego_lane, tgt_lane, self.lane_change_count, done, self.obs[self.STEPS_SINCE_LN_CHG]))

        # Update the obs vector with the new state info
        self.obs[self.EGO_ACCEL_CMD_PREV2] = self.obs[self.EGO_ACCEL_CMD_PREV1]
        self.obs[self.EGO_ACCEL_CMD_PREV1] = self.obs[self.EGO_ACCEL_CMD_CUR]
        self.obs[self.EGO_ACCEL_CMD_CUR] = 0 #TODO dummy
        self.obs[self.EGO_LANE_CMD_CUR] = lc_cmd - 1 #maps to [-1, 0, 1]
        self.obs[self.EGO_LANE_ID] = new_ego_lane
        self.obs[self.EGO_LANE_REM] = new_ego_rem
        self.obs[self.EGO_SPEED] = new_ego_speed
        self.obs[self.EGO_X] = new_ego_x
        self.obs[self.ADJ_LN_LEFT_ID] = lid
        self.obs[self.ADJ_LN_LEFT_CONN_A] = la
        self.obs[self.ADJ_LN_LEFT_CONN_B] = lb
        self.obs[self.ADJ_LN_LEFT_REM] = l_rem
        self.obs[self.ADJ_LN_RIGHT_ID] = rid
        self.obs[self.ADJ_LN_RIGHT_CONN_A] = ra
        self.obs[self.ADJ_LN_RIGHT_CONN_B] = rb
        self.obs[self.ADJ_LN_RIGHT_REM] = r_rem
        self._verify_obs_limits("step after updating obs vector")

        # Check that none of the vehicles have crashed into each other, accounting for a lane change in progress
        # taking up both lanes
        crash = False
        if not done:
            crash = self._check_for_collisions()
            done = crash
            if done:
                self.num_crashes += 1
                return_info["reason"] = "Crashed into neighbor vehicle"

        # If vehicle has been stopped for several time steps, then declare the episode done as a failure
        stopped_vehicle = False
        if self.vehicles[0].speed < 0.5:
            self.stopped_count += 1
            if self.stopped_count > 3:
                done = True
                stopped_vehicle = True
                return_info["reason"] = "Vehicle is crawling to a stop"
        else:
            self.stopped_count = 0

        # Determine the reward resulting from this time step's action
        reward, expl = self._get_reward(done, crash, ran_off_road, stopped_vehicle)
        return_info["reward_detail"] = expl

        # Verify that the obs are within design limits
        self._verify_obs_limits("step after reward calc")

        # According to gym docs, return tuple should have 5 elements:
        #   obs
        #   reward
        #   done
        #   truncated (bool) - this one appears to not be supported in gym 0.26.1
        #   info
        if self.debug > 0:
            print("///// step complete. Returning obs = ")
            print(      self.obs)
            print("      reward = ", reward, ", done = ", done)
            print("      final vehicles array =")
            for i, v in enumerate(self.vehicles):
                v.print(i)
            print("      reason = {}".format(return_info["reason"]))
            print("      reward_detail = {}".format(return_info["reward_detail"]))

        truncated = False #intended to indicate if the episode ended prematurely due to step/time limit
        return self.obs, reward, done, truncated, return_info


    def get_stopper(self):
        """Returns the stopper object."""
        return self.stopper


    def get_burn_in_iters(self):
        """Returns the number of burn-in iterations configured."""
        return self.burn_in_iters


    def get_total_steps(self):
        """Returns the total number of time steps executed so far."""
        return self.total_steps


    def get_vehicle_dist_downtrack(self,
                                   vehicle_id   : int   #index of the vehicle of interest
                                  ) -> float:
        """Returns the indicated vehicle's distance downtrack from its lane beginning, in m.
            Used for inference, which needs real DDT, not X location.
        """
        #TODO: reconcile graphics X vs roadway X, given that one is not on an angle, and apparent speeds are therefore different.

        assert 0 <= vehicle_id < 4, "///// SimpleHighwayRamp.get_vehicle_dist_downtrack: illegal vehicle_id entered: {}".format(vehicle_id)

        ddt = self.vehicles[vehicle_id].x
        lane_id = self.vehicles[vehicle_id].lane_id
        if lane_id == 2:
            ddt -= self.roadway.get_lane_start_x(lane_id)
        return ddt


    def close(self):
        """Closes any resources that may have been used during the simulation."""
        pass #this method not needed for this version


    def initialize_ramp_vehicle_speed(self,
                                      relative_pos    : int = 2,  #desired position of the ego vehicle relative to the 3 neighbors
                                                                  # at the time it reaches the merge area
                                      ego_x           : float = 0.0 #ego vehicle's X location, m
                                     ) -> float:
        """Returns a speed to start the ego vehicle when it starts at the beginning of lane 2, such that when it arrives at the
            beginning of the merge area, assuming it continues at the same speed, it will approximately match the specified
            position relative to the approaching neighbor vehicles.
            This location will be randomized as a normal draw near the specified relative position.  Relative position codes are:
                0 = in front of neighbor #1
                1 = roughly beside neighbor #1
                2 = roughly beside neighbor #2
                3 = roughly beside neighbor #3
                4 = behind neighbor #3
            Note that this method only makes sense in difficulty level 4, where the neighbors start lined up in lane 1 (neighbor 1
            in front and neighbor 3 in the rear), and they travel at constant speed with no lane changes.  So predicting their
            arrival at the merge area is trivial.  Estimating the ego vehicle's nominal arrival at the merge area assumes that it
            would try to achieve a max-reward trajectory, which is max accel until it hits speed limit, then stay at that speed.
        """

        L1_DIST_TO_MERGE = 800.0 #m, depends on code in Roadway class
        L2_DIST_TO_MERGE = 740.0 #m, depends on code in Roadway class
        headway = 3.0 * SimpleHighwayRamp.VEHICLE_LENGTH #this is hard-coded into the reset() method

        # Find the time of arrival of the target neighbor
        tgt_arrival_time = 0.0
        if relative_pos <= 1:
            tgt_arrival_time = (L1_DIST_TO_MERGE - self.neighbor_start_loc - 2*headway) / self.neighbor_speed
        elif relative_pos == 2:
            tgt_arrival_time = (L1_DIST_TO_MERGE - self.neighbor_start_loc - headway) / self.neighbor_speed
        else:
            tgt_arrival_time = (L1_DIST_TO_MERGE - self.neighbor_start_loc) / self.neighbor_speed
        #print("\n///// initialize_ramp_vehicle_speed: rel pos = {}, headway = {:.1f}, n start loc = {:.1f}, n spd = {:.1f}, tgt time = {:.1f}"
        #      .format(relative_pos, headway, self.neighbor_start_loc, self.neighbor_speed, tgt_arrival_time))

        # Get a random offset from that arrival time and apply it
        time_headway = headway / self.neighbor_speed
        offset = self.prng.normal(scale = 0.1*time_headway)
        if relative_pos == 0: #in front of first neighbor
            offset -= 1.1 * time_headway
        elif relative_pos == 4: #behind last neighbor
            offset += 1.1 * time_headway
        tgt_arrival_time += offset

        # If our desired arrival time is large enough then
        v0 = SimpleHighwayRamp.ROAD_SPEED_LIMIT
        #print("///// initialize_ramp_vehicle_speed: tgt time = {:.1f}, offset = {:.1f}, ego_x = {:.1f}".format(tgt_arrival_time, offset, ego_x))
        dist_to_merge = self.roadway.get_lane_start_x(2) + L2_DIST_TO_MERGE - ego_x
        if tgt_arrival_time > dist_to_merge/SimpleHighwayRamp.ROAD_SPEED_LIMIT:

            # Solve quadratic equation to determine the ramp vehicle's starting speed, assuming that it will
            # accelerate at the max rate until it reaches speed limit, then stays at that speed (this would maximize its reward for
            # the pre-merge part of the episode, so it will have a desire to stay close to this trajectory).
            vf = SimpleHighwayRamp.ROAD_SPEED_LIMIT
            qa = -0.5 / SimpleHighwayRamp.MAX_ACCEL
            qb = vf / SimpleHighwayRamp.MAX_ACCEL
            qc = vf*tgt_arrival_time - 0.5*vf*vf/SimpleHighwayRamp.MAX_ACCEL - dist_to_merge
            #print("///// initialize_ramp_vehicle_speed: qa = {:.4f}, qb = {:.4f}, qc = {:.4f}, tgt time = {:.1f}"
            #    .format(qa, qb, qc, tgt_arrival_time))
            root_part = math.sqrt(qb*qb - 4.0*qa*qc)
            v0 = (-qb + root_part) / 2.0 / qa
            if v0 <= 0.0  or  v0 > SimpleHighwayRamp.ROAD_SPEED_LIMIT:
                v0 = (-qb - root_part) / 2.0 / qa
                if v0 <= 0.0  or  v0 > SimpleHighwayRamp.ROAD_SPEED_LIMIT:
                    # Neither root works, which means target time is probably too long to be able to solve with max acceleration.
                    # So pick a random slow speed to at least get close to what is desired.
                    v0 = self.prng.random() * 0.3*SimpleHighwayRamp.ROAD_SPEED_LIMIT

            #print("///// initialize_ramp_vehicle_speed computed desired speed = {:.1f} for relative_pos = {}, tgt arrival = {:.1f} s"
            #    .format(v0, relative_pos, tgt_arrival_time))

        else: #ego will have to start faster than speed limit to hit the selected relative position
            v0 = self.prng.random() * (SimpleHighwayRamp.MAX_SPEED - SimpleHighwayRamp.ROAD_SPEED_LIMIT) + SimpleHighwayRamp.ROAD_SPEED_LIMIT
            #print("///// initialize_ramp_vehicle_speed: can't do quadratic. Choosing v0 = {:.1f}".format(v0))

        if v0 > SimpleHighwayRamp.MAX_SPEED:
            v0 = SimpleHighwayRamp.MAX_SPEED
        elif v0 <= 0.0:
            v0 = 1.3

        return v0


    ##### internal methods #####


    def _set_initial_conditions(self,
                                config:     EnvContext
                               ):
        """Sets the initial conditions of the ego vehicle in member variables (lane ID, speed, location)."""

        self.burn_in_iters = 0
        try:
            bi = config["burn_in_iters"]
            if bi > 0:
                self.burn_in_iters = int(bi)
        except KeyError as e:
            pass

        self.time_step_size = 0.5 #seconds
        try:
            ts = config["time_step_size"]
        except KeyError as e:
            ts = None
        if ts is not None  and  ts != ""  and  float(ts) > 0.0:
            self.time_step_size = float(ts)

        self.debug = 0
        try:
            db = config["debug"]
        except KeyError as e:
            db = None
        if db is not None  and  db != ""  and  0 <= int(db) <= 2:
            self.debug = int(db)

        self.verify_obs = False
        try:
            vo = config["verify_obs"]
            self.verify_obs = vo
        except KeyError as e:
            pass

        self.training = False
        try:
            tr = config["training"]
            if tr:
                self.training = True
        except KeyError as e:
            pass

        self.randomize_start_dist = False
        try:
            rsd = config["randomize_start_dist"]
            if rsd:
                self.randomize_start_dist = True
        except KeyError as e:
            pass

        self.init_ego_lane = None
        try:
            el = config["init_ego_lane"]
            if 0 <= el <= 2:
                self.init_ego_lane = el
        except KeyError as e:
            pass

        self.init_ego_speed = None
        try:
            es = config["init_ego_speed"]
            if 0 <= es <= SimpleHighwayRamp.MAX_SPEED:
                self.init_ego_speed = es
        except KeyError as e:
            pass

        self.init_ego_dist = None
        try:
            ed = config["init_ego_dist"]
            if 0 <= ex < SimpleHighwayRamp.SCENARIO_LENGTH:
                self.init_ego_dist = ed
        except KeyError as e:
            pass

        self.neighbor_speed = 29.1
        try:
            ns = config["neighbor_speed"]
            if 0 <= ns < SimpleHighwayRamp.MAX_SPEED:
                self.neighbor_speed = ns
        except KeyError as e:
            pass

        self.neighbor_start_loc = 0.0
        try:
            nsl = config["neighbor_start_loc"]
            if 0 <= nsl <= 1000.0:
                self.neighbor_start_loc = nsl
        except KeyError as e:
            pass

        self.merge_relative_pos = 2 #default to beside neighbor 2
        try:
            mrp = config["merge_relative_pos"]
            if 0 <= mrp <= 4:
                self.merge_relative_pos = mrp
        except KeyError as e:
            pass

        self.difficulty_level = 0
        try:
            dl = config["difficulty_level"]
            if 0 <= dl < SimpleHighwayRamp.NUM_DIFFICULTY_LEVELS:
                self.difficulty_level = int(dl)
        except KeyError as e:
            pass

        # Store the stopper, but also let the stopper know how to reach this object
        self.stopper = None
        try:
            s = config["stopper"]
            s.set_environment_model(self)
            self.stopper = s
        except KeyError as e:
            pass #print("///// INFO: Stopper not specified in environment config.")


    def _select_init_lane(self) -> int:
        """Chooses the initial lane for training runs, which may not be totally random."""

        # Levels 0-2 are restricted to lanes 0 & 1 (the full-length lanes)
        if self.difficulty_level < 3:
            return int(self.prng.random()*2) #select 0 or 1

        # Levels 3 & 4 need to emphasizes lots of experience in lane 2
        elif self.difficulty_level == 3:
            if self.prng.random() < 0.5:
                return 2
            else:
                return int(self.prng.random()*2) #select 0 or 1

        elif self.difficulty_level < 5:
            if self.prng.random() < 0.5:
                return 2
            else:
                return int(self.prng.random()*2) #select 0 or 1

        else:
            return int(self.prng.random()*3)


    def _check_for_collisions(self):
        """Compares location and bounding box of each vehicle with all other vehicles to determine if there are
            any overlaps.  If any two vehicle bounding boxes overlap, then returns True, otherwise False.

            Return: (bool) has there been a collision?
        """

        if self.debug > 1:
            print("///// Enteriing _check_for_collisions")
        crash = False

        # Loop through all vehicles to get vehicle A
        for i in range(len(self.vehicles) - 1):
            va = self.vehicles[i]

            # Loop through the remaining vehicles to get vehicle B
            for j in range(i + 1, len(self.vehicles)):
                vb = self.vehicles[j]

                # If A and B are in the same lane, then
                if va.lane_id == vb.lane_id:

                    # If they are within one car length of each other, it's a crash
                    if abs(va.x - vb.x) <= SimpleHighwayRamp.VEHICLE_LENGTH:
                        if self.debug > 1:
                            print("      CRASH in same lane between vehicles {} and {} near {:.2f} m in lane {}"
                                    .format(i, j, va.x, va.lane_id))
                        crash = True
                        break

                # Else if they are in adjacent lanes, then
                elif abs(va.lane_id - vb.lane_id) == 1:

                    # If either vehicle is changing lanes at the moment, then
                    if va.lane_change_status != "none"  or  vb.lane_change_status != "none":

                        # If the lane changer's target lane belongs to the other vehicle, then
                        va_tgt = self.roadway.get_target_lane(va.lane_id, va.lane_change_status, va.x)
                        vb_tgt = self.roadway.get_target_lane(vb.lane_id, vb.lane_change_status, vb.x)
                        if va_tgt == vb.lane_id  or  vb_tgt == va.lane_id:

                            # If the two are within a vehicle length of each other, then it's a crash
                            if abs(va.x - vb.x) <= SimpleHighwayRamp.VEHICLE_LENGTH:
                                if self.debug > 1:
                                    print("      CRASH in adjacent lanes between vehicles {} and {} near {:.2f} m in lane {}"
                                            .format(i, j, vb.x, va.lane_id))
                                crash = True
                                break

            if crash: #the previous break stmts only break out of the inner loop, so we need to break again
                break

        if self.debug > 0:
            print("///// _check_for_collisions complete. Returning ", crash)
        return crash


    def _get_reward(self,
                    done    : bool,         #is this the final step in the episode?
                    crash   : bool,         #did one or more of the vehicles crash into each other?
                    off_road: bool,         #did the ego vehicle run off the road?
                    stopped : bool          #has the vehicle come to a standstill?
                   ):
        """Returns the reward for the current time step (float).  The reward should be in [-1, 1] for any situation."""

        if self.debug > 1:
            print("///// Entering _get_reward. done = {}, crash = {}, off_road = {}".format(done, crash, off_road))
        reward = 0.0
        explanation = ""

        # If the episode is done then
        if done:

            # If there was a multi-car crash or off-roading (single-car crash) then set a penalty, larger for multi-car crash
            if crash:
                reward = -20.0
                explanation = "Crashed into a vehicle. "

            elif off_road:
                reward = -10.0
                explanation = "Ran off road. "

            # Else if the vehicle just stopped in the middle of the road then
            elif stopped:

                # Subtract a penalty for no movement (needs to be as severe as off-road)
                reward = -10.0
                explanation = "Vehicle stopped. "

            # Else (episode ended successfully)
            else:

                # If we are allowed to reward the completion bonus then add amount inversely proportional
                # to the length of the episode.
                if self.difficulty_level == 0:
                    reward = 10.0
                    explanation = "Successful episode!"

                else:
                    if self.reward_for_completion:
                        reward = min(max(10.0 - 0.05882*(self.steps_since_reset - 130), 0.0), 10.0)
                        explanation = "Successful episode! {} steps".format(self.steps_since_reset)
                    else:
                        explanation = "Completed episode, but no bonus due to rule violation."

        # Else, episode still underway
        else:

            if self.difficulty_level > 0:
                # If ego vehicle acceleration is jerky, then apply a penalty (worst case 0.003)
                jerk = (self.obs[self.EGO_ACCEL_CMD_CUR] - self.obs[self.EGO_ACCEL_CMD_PREV1]) / self.time_step_size
                penalty = 0.004 * jerk*jerk
                reward -= penalty
                if penalty > 0.0001:
                    explanation += "Jerk pen {:.4f}. ".format(penalty)

            # Penalty for deviating from roadway speed limit
            speed_mult = 0.5
            if self.difficulty_level == 1  or  self.difficulty_level == 2:
                speed_mult = 1.0

            norm_speed = self.obs[self.EGO_SPEED] / SimpleHighwayRamp.ROAD_SPEED_LIMIT #1.0 = speed limit
            diff = abs(norm_speed - 1.0)
            penalty = 0.0
            if diff > 0.02:
                penalty = speed_mult*(diff - 0.02)
                explanation += "spd pen {:.4f}".format(penalty)
            reward -= penalty

            # If a lane change was initiated, apply a penalty depending on how soon after the previous lane change
            if self.lane_change_count == 1:
                penalty = 0.1 + 0.01*(SimpleHighwayRamp.MAX_STEPS_SINCE_LC - self.obs[self.STEPS_SINCE_LN_CHG])
                reward -= penalty
                explanation += "Ln chg pen {:.4f}. ".format(penalty)

        if self.debug > 0:
            print("///// reward returning {:.4f} due to crash = {}, off_road = {}, stopped = {}"
                    .format(reward, crash, off_road, stopped))

        return reward, explanation


    def _verify_obs_limits(self,
                           tag      : str = ""  #optional explanation of where in the code this was called
                          ):
        """Checks that each element of the observation vector is within the limits of the observation space."""

        if not self.verify_obs:
            return

        lo = self.observation_space.low
        hi = self.observation_space.high

        try:
            for i in range(SimpleHighwayRamp.OBS_SIZE):
                assert lo[i] <= self.obs[i] <= hi[i], "\n///// obs[{}] value ({}) is outside bounds {} and {}" \
                                                        .format(i, self.obs[i], lo[i], hi[i])

        except AssertionError as e:
            print(e)
            print("///// Full obs vector content at: {}:".format(tag))
            for j in range(SimpleHighwayRamp.OBS_SIZE):
                print("      {:2d}: {}".format(j, self.obs[j]))


######################################################################################################
######################################################################################################


def curriculum_fn(train_results:        dict,           #current status of training progress
                  task_settable_env:    TaskSettableEnv,#the env model that difficulties will be applied to
                  env_ctx,                              #???
                 ) -> int:                              #returns the new difficulty level

    """Callback that allows Ray.Tune to increase the difficulty level, based on the current training results."""

    """NOTE: reconsider automated promotion logic. Skipping this logic allows manually setting
        the level as a constant for each Tune experiment. In this way, variable HPs, like annealing LR
        and noise levels can easily be set and scaled down through the course of a single level. It is
        not clear how to automate the resetting of these kinds of HPs.
    """

    return task_settable_env.get_task()

    # If the mean reward is above the success threshold for the current phase then advance the phase
    assert task_settable_env is not None, "\n///// Unable to access task_settable_env in curriculum_fn."
    phase = task_settable_env.get_task()
    #stopper = task_settable_env.get_stopper()
    #assert stopper is not None, "\n///// Unable to access the stopper object in curriculum_fn."
    total_steps_sampled = task_settable_env.get_total_steps() #resets after a perturbation

    #value = train_results["episode_reward_mean"]
    #burn_in = task_settable_env.get_burn_in_iters()
    #approx_steps_per_iter = 300.0 #for an individual environment, not across all workers
    #iter = int(total_steps_sampled / approx_steps_per_iter)
    #print("///// curriculum_fn: phase = {}, value = {}, approx_steps_per_iter = {}, iter = {}".format(phase, value, approx_steps_per_iter, iter))
    #print("/////                results num_env_steps_sampled = {}, env total steps = {}"
    #      .format(total_steps_sampled, task_settable_env.get_total_steps()))

    # If there has been no perturbations performed yet (still prior to the first cycle) then
    TARGET_PHASE = 3
    if not perturb_ctrl.has_perturb_begun():

        # When threshold number of steps expires, advance from phase 0 to phase 4; this allows the agent to start
        # figuring out where the finish line is, but otherwise do all training at the phase 4 difficulty.
        if phase < TARGET_PHASE  and  total_steps_sampled > 70000: #MUST OCCUR PRIOR TO FIRST PERTURB CYCLE
            print("///// curriculum_fn advancing phase from {} to {}".format(phase, TARGET_PHASE))
            task_settable_env.set_task(TARGET_PHASE)

    # Else go straight to the target phase
    else:
        if phase != TARGET_PHASE:
            task_settable_env.set_task(TARGET_PHASE)

    """This section is for normal automated curriculum progression
    if iter >= burn_in  and  value >= stopper.get_success_thresholds()[phase]:
        print("\n///// curriculum_fn in phase {}: iter = {}, burn-in = {}, episode_reward_mean = {}"
              .format(phase, iter, burn_in, value))
        #task_settable_env.set_task(phase+1) #TODO uncomment to change phases
    """

    return task_settable_env.get_task() #don't return the local one, since the env may override it


######################################################################################################
######################################################################################################



class Roadway:
    """Defines the geometry of the roadway lanes and their drivable connections.  All dimensions are
        physical quantities, measured in meters from an arbitrary map origin.  The roadway being modeled
        looks roughly like the diagram at the top of this code file.  However, this class abstracts it
        slightly to be more of a structural "schematic" for better representation in our NN observation
        space. To that end, all lanes in the schematic are considered parallel and physically next to
        each other (even though the boundary separating two lanes may not be permeable, e.g. a jersey
        wall).

        All lanes go from left to right, with travel direction being to the right. The coordinate system
        is oriented so that the origin is at the left (beginning of the first lane), with the X axis
        going to the right and Y axis going upward on the page. Not all lanes have to begin at X = 0,
        but at least one does. Others may begin at X > 0. Y locations and the lane segments are only used
        for the graphics output; they are not needed for the environment calculations, per se.

        CAUTION: This is not a general container.  This __init__ code defines the exact geometry of the
        scenario being used by this version of the code.
    """

    WIDTH = 20.0 #lane width, m; using a crazy large number so that grapics are pleasing

    def __init__(self,
                 debug      : int   #debug printing level
                ):

        self.debug = debug
        if self.debug > 1:
            print("///// Entering Roadway.__init__")
        self.lanes = [] #list of all the lanes in the scenario; list index is lane ID

        # Full length of the modeled lane, extends beyond the length of the scenario so the agent
        # views the road as a continuing situation, rather than an episodic game.
        really_long = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH

        # Lane 0 - single segment as the left through lane
        L0_Y = 300.0 #arbitrary y value for the east-bound lane
        segs = [(0.0, L0_Y, really_long, L0_Y, really_long)]
        lane = Lane(0, 0.0, really_long, segs,
                    right_id = 1, right_join = 0.0, right_sep = really_long)
        self.lanes.append(lane)

        # Lane 1 - single segment as the right through lane
        L1_Y = L0_Y - Roadway.WIDTH
        segs = [(0.0, L1_Y, really_long, L1_Y, really_long)]
        lane = Lane(1, 0.0, really_long, segs,
                    left_id = 0, left_join = 0.0, left_sep = really_long,
                    right_id = 2, right_join = 800.0, right_sep = 1320.0)
        self.lanes.append(lane)

        # Lane 2 - two segments as the merge ramp; first seg is separate; second it adjacent to L1.
        # Segments show the lane at an angle to the main roadway, for visual appeal & clarity; this is
        # different from how it is represented internally (parallel to the other lanes).
        L2_Y = L1_Y - Roadway.WIDTH
        segs = [(159.1, L2_Y-370.0,  800.0, L2_Y, 740.0),
                (800.0, L2_Y,       1320.0, L2_Y, 520.0)]
        lane = Lane(2, 60.0, 1260.0, segs, left_id = 1, left_join = 800.0, left_sep = 1320.0)
        self.lanes.append(lane)


    def get_current_lane_geom(self,
                                lane_id         : int   = 0,    #ID of the lane in question
                                x_loc           : float = 0.0   #ego vehicle's location in the X coordinate, m
                             ) -> Tuple[float, int, float, float, float, int, float, float, float]:
        """Determines all of the variable roadway geometry relative to the given vehicle location.
            Returns a tuple of (remaining dist in this lane,
                                ID of left neighbor ln (or -1 if none),
                                dist to left adjoin point A,
                                dist to left adjoin point B,
                                remaining dist in left ajoining lane,
                                ID of right neighbor lane (or -1 if none),
                                dist to right adjoin point A,
                                dist to right adjoin point B,
                                remaining dist in right adjoining lane).
            If either adjoining lane doesn't exist, its return values will be 0, inf, inf, inf.  All distances are in m.
        """

        if self.debug > 1:
            print("///// Entering Roadway.get_current_lane_geom for lane_id = ", lane_id, ", x_loc = ", x_loc)
        rem_this_lane = self.lanes[lane_id].start_x + self.lanes[lane_id].length - x_loc

        la = 0.0
        lb = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        l_rem = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        left_id = self.lanes[lane_id].left_id
        if left_id >= 0:
            la = self.lanes[lane_id].left_join - x_loc
            lb = self.lanes[lane_id].left_sep - x_loc
            l_rem = self.lanes[left_id].start_x + self.lanes[left_id].length - x_loc

        ra = 0.0
        rb = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        r_rem = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        right_id = self.lanes[lane_id].right_id
        if right_id >= 0:
            ra = self.lanes[lane_id].right_join - x_loc
            rb = self.lanes[lane_id].right_sep - x_loc
            r_rem = self.lanes[right_id].start_x + self.lanes[right_id].length - x_loc

        if self.debug > 0:
            print("///// get_current_lane_geom complete. Returning rem = ", rem_this_lane)
            print("      lid = {}, la = {:.2f}, lb = {:.2f}, l_rem = {:.2f}".format(left_id, la, lb, l_rem))
            print("      rid = {}, ra = {:.2f}, rb = {:.2f}, r_rem = {:.2f}".format(right_id, ra, rb, r_rem))
        return rem_this_lane, left_id, la, lb, l_rem, right_id, ra, rb, r_rem


    def adjust_downtrack_dist(self,
                                prev_lane_id    : int,
                                new_lane_id     : int
                             ) -> float:
        """NO-OP - replacing deprecated method that was used when downtrack distance was measured from lane start location,
            which could be different for each lane. We are now using x coordinate instead of downtrack dist, which is
            a common reference, regardless of lane geometry.

        Returns an adjustment to be applied to the downtrack distance in the current lane, as a result of changing lanes.
            A vehicle's downtrack distance is relative to the beginning of its current lane, and each lane may start at a
            different point.
            Return value can be positive or negative (float), m
        """

        #TODO: remove this method once new geometry is fully tested
        return 0.0


    def get_target_lane(self,
                        lane        : int,  #ID of the given lane
                        direction   : str,  #either "left" or "right"
                        distance    : float #X location of interest, m
                       ) -> int:
        """Returns the lane ID of the adjacent lane on the indicated side of the given lane, or -1 if there is none
            currently adjoining.
        """

        if self.debug > 1:
            print("///// Entering Roadway.get_target_lane. lane = ", lane, ", direction = ", direction, ", distance = ", distance)
        assert 0 <= lane < len(self.lanes), "get_adjoining_lane_id input lane ID {} invalid.".format(lane)
        if direction != "left"  and  direction != "right":
            return -1

        # Find the adjacent lane ID, then if one exists ensure that current location is between the join & separation points.
        this_lane = self.lanes[lane]
        if direction == "left":
            tgt = this_lane.left_id
            if tgt >= 0:
                if distance < this_lane.left_join  or  distance > this_lane.left_sep:
                    tgt = -1

        else: #right
            tgt = this_lane.right_id
            if tgt >= 0:
                if distance < this_lane.right_join  or  distance > this_lane.right_sep:
                    tgt = -1

        if self.debug > 1:
            print("///// get_target_lane complete. Returning ", tgt)
        return tgt


    def get_total_lane_length(self,
                                lane    : int   #ID of the lane in question
                             ) -> float:
        """Returns the total length of the requested lane, m"""

        assert 0 <= lane < len(self.lanes), "Roadway.get_total_lane_length input lane ID {} invalid.".format(lane)
        return self.lanes[lane].length


    def get_lane_start_x(self,
                         lane   : int   #ID of the lane in question
                        ) -> float:
        """Returns the X coordinate of the beginning of the lane."""

        assert 0 <= lane < len(self.lanes), "Roadway.get_lane_start_x input lane ID {} invalid.".format(lane)
        return self.lanes[lane].start_x



######################################################################################################
######################################################################################################



class Lane:
    """Defines the geometry of a single lane and its relationship to other lanes.
        Note: an adjoining lane must join this one exactly once (possibly at distance 0), and
                may or may not separate from it farther downtrack. Once it separates, it cannot
                rejoin.  If it does not separate, then separation location will be same as this
                lane's length.
    """

    def __init__(self,
                    my_id       : int,                  #ID of this lane
                    start_x     : float,                #X coordinate of the start of the lane, m
                    length      : float,                #total length of this lane, m (includes buffer)
                    segments    : list,                 #list of straight segments that make up this lane; each item is
                                                        # a tuple containing: (x0, y0, x1, y1, length), where
                                                        # x0, y0 are map coordinates of the starting point, in m
                                                        # x1, y1 are map coordinates of the ending point, in m
                                                        # length is the length of the segment, in m
                                                        #Each lane must have at least one segment, and segment lengths
                                                        # need to add up to total lane length
                    left_id     : int       = -1,       #ID of an adjoining lane to its left, or -1 for none
                    left_join   : float     = 0.0,      #X location where left lane first joins, m
                    left_sep    : float     = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH,
                                                        #X location where left lane separates from this one, m
                    right_id    : int       = -1,       #ID of an ajoining lane to its right, or -1 for none
                    right_join  : float     = 0.0,      #X location where right lane first joins, m
                    right_sep   : float     = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
                                                        #X location where right lane separates from this one, m
                ):

        self.my_id = my_id
        self.start_x = start_x
        self.length = length
        self.left_id = left_id
        self.left_join = left_join
        self.left_sep = left_sep
        self.right_id = right_id
        self.right_join = right_join
        self.right_sep = right_sep
        self.segments = segments

        assert start_x >= 0.0, "Lane {} start_x {} is invalid.".format(my_id, start_x)
        assert length > 0.0, "Lane {} length of {} is invalid.".format(my_id, length)
        assert left_id >= 0  or  right_id >= 0, "Lane {} has no adjoining lanes.".format(my_id)
        assert len(segments) > 0, "Lane {} has no segments defined.".format(my_id)
        seg_len = 0.0
        for si, s in enumerate(segments):
            seg_len += s[4]
            assert s[0] != s[2]  or  s[1] != s[3], "Lane {}, segment {} both ends have same coords.".format(my_id, si)
        assert abs(seg_len - length) < 1.0, "Lane {} sum of segment lengths {} don't match total lane length {}.".format(my_id, seg_len, length)
        if left_id >= 0:
            assert left_id != my_id, "Lane {} left adjoining lane has same ID".format(my_id)
            assert left_join >= 0.0  and  left_join < length+start_x, "Lane {} left_join value invalid.".format(my_id)
            assert left_sep > left_join, "Lane {} left_sep {} not larger than left_join {}".format(my_id, left_sep, left_join)
            assert left_sep <= length+start_x, "Lane {} left sep {} is beyond end of lane.".format(my_id, left_sep)
        if right_id >= 0:
            assert right_id != my_id, "Lane {} right adjoining lane has same ID".format(my_id)
            assert right_join >= 0.0  and  right_join < length+start_x, "Lane {} right_join value invalid.".format(my_id)
            assert right_sep > right_join, "Lane {} right_sep {} not larger than right_join {}".format(my_id, right_sep, right_join)
            assert right_sep <= length+start_x, "Lane {} right sep {} is beyond end of lane.".format(my_id, right_sep)
        if left_id >= 0  and  right_id >= 0:
            assert left_id != right_id, "Lane {}: both left and right adjoining lanes share ID {}".format(my_id, left_id)



######################################################################################################
######################################################################################################



class Vehicle:
    """Represents a single vehicle on the Roadway."""

    def __init__(self,
                    step_size   : float,    #duration of a time step, s
                    max_jerk    : float,    #max allowed jerk, m/s^3
                    debug       : int       #debug printing level
                ):

        self.time_step_size = step_size
        self.max_jerk = max_jerk
        self.debug = debug

        self.lane_id = -1                   #-1 is an illegal value
        self.x = 0.0                        #X coordinate of vehicle center on map, m
        self.lane_change_status = "none"    #Initialized to no lane change underway
        self.speed = 0.0                    #Forward speed, m/s


    def advance_vehicle_spd(self,
                        new_speed_cmd   : float,    #the newly commanded speed, m/s
                        cur_speed       : float,    #the current actual speed, m/s
                        prev_speed      : float     #actual speed at the previous time step, m/s
                       ) -> Tuple[float, float]:
        """Advances a vehicle's forward motion for one time step according to the vehicle dynamics model.
            Note that this does not consider lateral motion, which needs to be handled elsewhere.

            Returns: tuple of (new speed (m/s), new X location (m))
        """

        # Determine the current & previous effective accelerations
        cur_accel_cmd = (new_speed_cmd - cur_speed) / self.time_step_size
        prev_accel = (cur_speed - prev_speed) / self.time_step_size
        print("///// Vehicle.advance_vehicle_spd: new_speed_cmd = {:.1f}, cur_speed = {:.1f}, prev_speed = {:.1f}, cur_accel_cmd = {:.2f}, prev_accel = {:.2f}"
              .format(new_speed_cmd, cur_speed, prev_speed, cur_accel_cmd, prev_accel))
        return self.advance_vehicle_acc(cur_accel_cmd, prev_accel)


    def advance_vehicle_acc(self,
                        new_accel_cmd   : float,    #newest fwd accel command, m/s^2
                        prev_accel_cmd  : float     #fwd accel command from prev time step, m/s^2
                       ) -> Tuple[float, float]:
        """Advances a vehicle's forward motion for one time step according to the vehicle dynamics model.
            Note that this does not consider lateral motion, which needs to be handled elsewhere.

            Returns: tuple of (new speed (m/s), new X location (m))
        """

        # Determine new jerk, accel, speed & location of the vehicle
        new_jerk = (new_accel_cmd - prev_accel_cmd) / self.time_step_size
        if new_jerk < -self.max_jerk:
            new_jerk = -self.max_jerk
        elif new_jerk > self.max_jerk:
            new_jerk = self.max_jerk
        print("///// Vehicle.advance_vehicle_acc: new_jerk = {:.3f}".format(new_jerk))

        new_accel = min(max(prev_accel_cmd + self.time_step_size*new_jerk, -SimpleHighwayRamp.MAX_ACCEL), SimpleHighwayRamp.MAX_ACCEL)
        new_speed = min(max(self.speed + self.time_step_size*new_accel, 0.0), SimpleHighwayRamp.MAX_SPEED) #vehicle won't start moving backwards
        new_x = max(self.x + self.time_step_size*(new_speed + 0.5*self.time_step_size*new_accel), 0.0)

        self.x = new_x
        self.speed = new_speed

        return new_speed, new_x


    def print(self,
                tag     : object = None     #tag to identify the vehicle
             ):
        """Prints the attributes of this vehicle object."""

        print("       [{}]: lane_id = {:2d}, dist = {:.2f}, status = {:5s}, speed = {:.2f}" \
                .format(tag, self.lane_id, self.x, self.lane_change_status, self.speed))
