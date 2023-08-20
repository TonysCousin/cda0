from collections import deque
from statistics import mean
from typing import Tuple, Dict, List
import math
from datetime import datetime
from gymnasium.spaces import Box
import numpy as np
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.tune.logger import pretty_print
from hp_prng import HpPrng

# Define a perturbation controller here, it is to be used - needs to be outside of the class. Needed for restarting
# training from a checkpoint.
perturb_ctrl = None


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

        This simple enviroment assumes perfect traction and steering response, so that the physics of driving can
        essentially be ignored.  However, there are upper and lower limits on longitudinal and lateral acceleration
        capabilities of each vehicle.

        We will assume that all cars follow their chosen lane's centerlines at all times, except when
        changing lanes.  So there is no steering action required, per se, but there may be opportunity to change lanes
        left or right at certain times.  To support this simplification, the ramp will not join the mainline lane
        directly (e.g. distance between centerlines gradually decreases to 0).  Rather, it will approach asymptotically,
        then run parallel to the mainline lane for some length, then suddenly terminate.  It is in this parallel section
        that a car is able to change from the ramp lane to the mainline lane.  If it does not, it will be considered to
        have run off the end of the ramp pavement.  The geometry looks something like below, with lanes 0
        and 1 continuing indefinitely, while lane 2 (the on ramp) has a finite length.  Vehicle motion is generally
        from left to right.

            Lane 0  ------------------------------------------------------------------------------------------>
            Lane 1  ------------------------------------------------------------------------------------------>
                                                     ----------------------/
                                                    /
                                                   /
                                                  /
                    Lane 2  ----------------------

        The environment is a continuous flat planar space in the map coordinate frame, with the X-axis origin at the
        left end of lanes 0 & 1. The location of any vehicle is represented by its X value and lane ID (which constrains
        Y), so the Y origin is arbitrary, and only used for graphical output.

        In order to support data transforms to/from a neural network, the geometry is easier modeled schematically, where
        all lanes are parallel and "adjacent". But any given lane may not always be reachable (via lane change) from an
        adjacent lane (e.g. due to a jersey wall or grass in between the physical lanes). We call this schematic
        representation the parametric coordinate frame, which uses coordinates P and Q as analogies to X and Y in the map
        frame. In practice, the Q coordinate is never needed, since it is implied by the lane ID, which is more readily
        available and important. It is possible to use this parametric frame because we ASSUME that the vehicles are
        generally docile and not operating at their physical performance limits (acceleration, traction), and therefore
        can perform any requested action regardless of the shape of the lane at that point. The roadway shown above then
        looks something like the following in the parametric frame. It preserves the length of each lane segment. Note
        that the boundary between lane 1 and the first part of lane 2 is not permeable (representing the grass that is
        between these in reality).

            Lane 0  ------------------------------------------------------------------------------------------>
            Lane 1  ------------------------------------------------------------------------------------------>
                 Lane 2  ============================----------------------/

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
            5 = level 3 plus 3 randomly located, coonstant-speed vehicles with ACC anywhere on the track

        Agent rewards are provided by a separate reward function.  The reward logic is documented there.
    """

    metadata = {"render_modes": None}
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    VEHICLE_LENGTH          = 20.0      #m
    NUM_LANES               = 3         #total number of unique lanes in the scenario
    MAX_SPEED               = 36.0      #vehicle's max achievable speed, m/s
    MAX_ACCEL               = 3.0       #vehicle's max achievable acceleration (fwd or backward), m/s^2
    MAX_JERK                = 4.0       #max desirable jerk for occupant comfort, m/s^3
    ROAD_SPEED_LIMIT        = 29.1      #Roadway's legal speed limit on all lanes, m/s (29.1 m/s = 65 mph)
    SCENARIO_LENGTH         = 2000.0    #total length of the roadway, m
    SCENARIO_BUFFER_LENGTH  = 200.0     #length of buffer added to the end of continuing lanes, m
    NUM_NEIGHBORS           = 6         #total number of neighbor vehicles in scenario (some or all may not be active)
    OBS_ZONE_LENGTH         = 2.0 * ROAD_SPEED_LIMIT #the length of a roadway observation zone, m
    #TODO: make this dependent upon time step size:
    HALF_LANE_CHANGE_STEPS  = 3.0 / 0.5 // 2    #num steps to get half way across the lane (equally straddling both)
    TOTAL_LANE_CHANGE_STEPS = 2 * HALF_LANE_CHANGE_STEPS
    MAX_STEPS_SINCE_LC      = 60        #largest num time steps we will track since previous lane change
    NUM_DIFFICULTY_LEVELS   = 6         #num levels of environment difficulty for the agent to learn; see descrip above
    # The following are for level neighbor adaptive cruise control (ACC) functionality
    DISTANCE_OF_CONCERN     = 8.0 * VEHICLE_LENGTH #following distance below which the vehicle needs to start slowing to avoid forward neighbor
    CRITICAL_DISTANCE       = 2.0 * VEHICLE_LENGTH #following distance below which the vehicle needs to be matching its forward neighbor's speed


    def __init__(self,
                 config:        EnvContext,             #dict of config params
                 seed:          int             = None, #seed for PRNG
                 render_mode:   int             = None  #Ray rendering info, unused in this version
                ):

        """Wrapper for the real initialization in order to trap stray exceptions."""

        try:
            self._init(config, seed, render_mode)
        except Exception as e:
            print("\n///// Exception trapped in SimpleHighwayRamp.__init__: ", e)


    def _init(self,
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
        if seed is None:
            seed = datetime.now().microsecond
        self.prng = HpPrng(seed = seed)
        self.render_mode = render_mode

        self._set_initial_conditions(config)
        if self.debug > 0:
            print("\n///// SimpleHighwayRamp init: config = ", config)

        # Define the vehicles used in this scenario - the ego vehicle (where the agent lives) is index 0
        self.vehicles = []
        for i in range(SimpleHighwayRamp.NUM_NEIGHBORS + 1):
            v = Vehicle(self.time_step_size, SimpleHighwayRamp.MAX_JERK, debug = self.debug)
            self.vehicles.append(v)

        #
        #..........Define the observation space
        #
        # A key portion of the obs space is a representation of spatial zones around the ego vehicle. These zones
        # move with the vehicle, and are a schematic representation of the nearby roadway situation. That is, they
        # don't include any lane shape geometry, representing every lane as straight, and physically adjacent to its
        # next lane. This is possible because we ASSUME that the vehicles are operating well inside their performance
        # limits, so that road geometry won't affect their ability to change lanes, accelerate or decelerate at any
        # desired rate. The zones are arranged as in the following diagram. Given this assumption, the observation
        # space can be defined in the parametric coordinate frame.
        #
        #              +----------+----------+----------+
        #              |  zone 7  |  zone 4  |  zone 1  |
        #   +----------+----------+----------+----------+
        #   |  zone 8  |  ego veh |  zone 5  |  zone 2  | >>>>> direction of travel
        #   +----------+----------+----------+----------+
        #              |  zone 9  |  zone 6  |  zone 3  |
        #              +----------+----------+----------+
        #
        # All zones are the same size, with their width equal to the lane width. Zone length is nominally 2 sec of
        # travel distance (at the posted speed limit). The ego vehicle is always centered in its zone, longitudinally.
        # If a neighbor vehicle is in the process of changing lanes, it will be observed to be in both adjacent zones through
        # the full lane change process. However, since the grid is centered on the ego vehicle, that vehicle will be
        # handled specially when it comes to lane change maneuvers, looking into either zone 7 or zone 9 concerning a
        # possible crash.
        #
        # Each zone will provide the following set of information:
        #   Is it drivable? E.g. is there a lane in that relative position
        #   Is the zone reachable from ego lane? I.e. lateral motion is possible and legal along the full length of the zone.
        #   Is there a neighbor vehicle in the zone? No more than one vehicle will be allowed in any given zone.
        #   Occupant's P location within the zone, if occupant exists ((P - Prear) / zone length), in [0, 1]
        #   Occupant's speed relative to ego vehicle, if occupant exists (delta-S / speed limit), in approx [-1.2, 1.2]

        # Indices into the observation vector
        self.UNUSED             =  0 ### index 0 is UNUSED - should always contain value 0.0
        self.UNUSED             =  1 ### INDEX 1 IS unused - should always contain value 0.0
        self.EGO_LANE_REM       =  2 #distance remaining in the agent's current lane, m     #TODO: is this redundant with zone info?
        self.EGO_SPEED          =  3 #agent's actual forward speed, m/s
        self.EGO_SPEED_PREV     =  4 #agent's actual speed in previous time step, m/s
        self.STEPS_SINCE_LN_CHG =  5 #num time steps since the previous lane change was initiated
        self.NEIGHBOR_IN_EGO_ZONE = 6 #if any neighbor vehicle is in the ego zone, then this value will be 1 if it is in front of the
                                     # ego vehicle or -1 if it is behind the ego vehicle. If no neigbhor is present, value = 0
        self.EGO_DES_ACCEL      =  7 #agent's most recent acceleration command, m/s (action feedback from this step)
        self.EGO_DES_ACCEL_PREV =  8 #desired acceleration from previous time step, m/s
        self.LC_CMD             =  9 #agent's most recent lane change command, quantized (values map to the enum class LaneChange)
        self.LC_CMD_PREV        = 10 #lane change command from previous time step, quantized

        self.Z1_DRIVEABLE       = 11 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z1_REACHABLE       = 12 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z1_OCCUPIED        = 13 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z1_ZONE_P          = 14 #occupant's P location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z1_REL_SPEED       = 15 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z2_DRIVEABLE       = 16 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z2_REACHABLE       = 17 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z2_OCCUPIED        = 18 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z2_ZONE_P          = 19 #occupant's P location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z2_REL_SPEED       = 20 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z3_DRIVEABLE       = 21 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z3_REACHABLE       = 22 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z3_OCCUPIED        = 23 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z3_ZONE_P          = 24 #occupant's P location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z3_REL_SPEED       = 25 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z4_DRIVEABLE       = 26 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z4_REACHABLE       = 27 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z4_OCCUPIED        = 28 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z4_ZONE_P          = 29 #occupant's P location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z4_REL_SPEED       = 30 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z5_DRIVEABLE       = 31 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z5_REACHABLE       = 32 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z5_OCCUPIED        = 33 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z5_ZONE_P          = 34 #occupant's P location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z5_REL_SPEED       = 35 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z6_DRIVEABLE       = 36 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z6_REACHABLE       = 37 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z6_OCCUPIED        = 38 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z6_ZONE_P          = 39 #occupant's P location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z6_REL_SPEED       = 40 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z7_DRIVEABLE       = 41 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z7_REACHABLE       = 42 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z7_OCCUPIED        = 43 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z7_ZONE_P          = 44 #occupant's P location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z7_REL_SPEED       = 45 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z8_DRIVEABLE       = 46 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z8_REACHABLE       = 47 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z8_OCCUPIED        = 48 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z8_ZONE_P          = 49 #occupant's P location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z8_REL_SPEED       = 50 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.Z9_DRIVEABLE       = 51 #is all of this zone drivable pavement? (bool 0 or 1)
        self.Z9_REACHABLE       = 52 #is all of this zone reachable from ego's lane? (bool 0 or 1)
        self.Z9_OCCUPIED        = 53 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
        self.Z9_ZONE_P          = 54 #occupant's P location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
        self.Z9_REL_SPEED       = 55 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)

        self.OBS_SIZE = self.Z9_REL_SPEED + 1 #number of elements in the observation vector

        # Gymnasium requires a member variable named observation_space. Since we are dealing with world scale values here, we
        # will need a wrapper to scale the observations for NN input. That wrapper will also need to use self.observation_space.
        # So here we must anticipate that scaling and leave the limits open enough to accommodate it.
        max_rel_speed = SimpleHighwayRamp.MAX_SPEED / SimpleHighwayRamp.ROAD_SPEED_LIMIT
        lower_obs = np.zeros((self.OBS_SIZE)) #most values are 0, so only the others are explicitly set below
        lower_obs[self.EGO_DES_ACCEL]       = -SimpleHighwayRamp.MAX_ACCEL
        lower_obs[self.EGO_DES_ACCEL_PREV]  = -SimpleHighwayRamp.MAX_ACCEL
        lower_obs[self.LC_CMD]              = LaneChange.CHANGE_LEFT
        lower_obs[self.LC_CMD_PREV]         = LaneChange.CHANGE_LEFT
        lower_obs[self.NEIGHBOR_IN_EGO_ZONE]= -1.0
        lower_obs[self.Z1_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z2_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z3_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z4_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z5_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z6_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z7_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z8_REL_SPEED]        = -max_rel_speed
        lower_obs[self.Z9_REL_SPEED]        = -max_rel_speed

        upper_obs = np.ones(self.OBS_SIZE) #most values are 1
        upper_obs[self.EGO_SPEED]           = SimpleHighwayRamp.MAX_SPEED
        upper_obs[self.EGO_SPEED_PREV]      = SimpleHighwayRamp.MAX_SPEED
        upper_obs[self.EGO_DES_ACCEL]       = SimpleHighwayRamp.MAX_ACCEL
        upper_obs[self.EGO_DES_ACCEL_PREV]  = SimpleHighwayRamp.MAX_ACCEL
        upper_obs[self.EGO_LANE_REM]        = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        upper_obs[self.LC_CMD]              = LaneChange.CHANGE_RIGHT
        upper_obs[self.LC_CMD_PREV]         = LaneChange.CHANGE_RIGHT
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

        self.observation_space = Box(low = lower_obs, high = upper_obs, dtype = float)
        if self.debug == 2:
            print("///// observation_space = ", self.observation_space)

        self.obs = np.zeros(self.OBS_SIZE) #will be returned from reset() and step()
        self._verify_obs_limits("init after space defined")

        #
        #..........Define the action space
        #

        # Specify these for what the NN will deliver, not world scale
        lower_action = np.array([-1.0, -1.0])
        upper_action = np.array([ 1.0,  1.0])
        self.action_space = Box(low=lower_action, high = upper_action, dtype = float)
        if self.debug == 2:
            print("///// action_space = ", self.action_space)

        #
        #..........Remaining initializations
        #

        # Create the roadway geometry
        self.roadway = Roadway(self.debug)

        # Other persistent data
        self.lane_change_count = 0  #num consecutive time steps since a lane change was begun
        self.total_steps = 0        #num time steps for this trial (worker), across all episodes; NOTE that this is different from the
                                    # total steps reported by Ray tune, which is accumulated over all rollout workers
        self.steps_since_reset = 0  #length of the current episode in time steps
        self.stopped_count = 0      #num consecutive time steps in an episode where vehicle speed is almost zero
        self.reward_for_completion = True #should we award the episode completion bonus?
        self.episode_count = 0      #number of training episodes (number of calls to reset())
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
                 task:      int         #ID of the task (lesson difficulty) to simulate, in [0, n)
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

        """Wrapper around the real reset method to trap for unhandled exceptions."""

        try:
            return self._reset(seed = seed, options = options)
        except Exception as e:
            print("\n///// Exception trapped in SimpleHighwayRamp.reset: ", e)
            return (None, None)


    def _reset(self, *,
              seed:         int             = None,    #reserved for a future version of gym.Environment
              options:      dict            = None
             ) -> Tuple[np.array, dict]:

        """Reinitializes the environment to prepare for a new episode.  This must be called before
            making any calls to step().

            Return tuple includes an array of observation values, plus a dict of additional info key-value
            pairs.

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

        # Redefine the vehicle data structures - the ego vehicle is index 0
        self.vehicles = []
        for i in range(SimpleHighwayRamp.NUM_NEIGHBORS + 1):
            v = Vehicle(self.time_step_size, SimpleHighwayRamp.MAX_JERK, debug = self.debug)
            self.vehicles.append(v)

        #
        #..........Initialize ego vehicle for training
        #

        # If the agent is being trained then
        ego_lane_id = None
        ego_p = None
        ego_speed = None
        max_ego_dist = 1.0
        if self.training:
            ego_lane_id = self._select_init_lane() #covers all difficulty levels
            ego_lane_start = self.roadway.get_lane_start_p(ego_lane_id)

            # If we are using a perturbation controller, then get its read on perturbations underway
            has_perturb_begun = False
            if perturb_ctrl is not None:
                has_perturb_begun = perturb_ctrl.has_perturb_begin()

            # If we are at difficulty level 0-3, then choose widely randomized initial conditions
            if self.difficulty_level < 4:
                ego_p = self.prng.random() * 3.0*SimpleHighwayRamp.VEHICLE_LENGTH + ego_lane_start
                if self.randomize_start_dist  and  not has_perturb_begun:
                    physical_limit = min(self.roadway.get_total_lane_length(ego_lane_id), SimpleHighwayRamp.SCENARIO_LENGTH) - 10.0
                    initial_steps = 6000000 #num steps to wait before starting to shrink the max distance
                    if self.total_steps <= initial_steps:
                        max_ego_dist = physical_limit
                    else:
                        max_ego_dist = max((self.total_steps - initial_steps) * (10.0 - physical_limit)/(1e6 - initial_steps) + physical_limit,
                                        10.0) #decreases over time steps
                    ego_p = self.prng.random() * max_ego_dist + ego_lane_start
                    if ego_lane_id == 1:    #TODO: for testing lane 2 learning only!
                        ego_p = self.prng.random() * (max_ego_dist - 800.0) + 800.0
                ego_speed = self.prng.random() * 5.0 + 30.0 #value in [5, 35] m/s

            elif self.difficulty_level == 4  or  self.difficulty_level == 5:
                # Code block inspired from above to support training jigher levels from scratch - agent needs to learn how to find the finish line,
                # so allow it to start anywhere along the route, sometimes very close to the finish line, then gradually force it uptrack.
                INITIAL_STEPS   = 0 #num steps to wait before starting to shrink the max distance
                FINAL_STEPS     = 1000000 #num steps where max distance reduction ends
                physical_limit = min(self.roadway.get_total_lane_length(ego_lane_id), SimpleHighwayRamp.SCENARIO_LENGTH) - 10.0
                ego_p = self.prng.random() * 3.0*SimpleHighwayRamp.VEHICLE_LENGTH + ego_lane_start
                if self.randomize_start_dist  and  not has_perturb_begun:
                    max_ego_dist = physical_limit
                    if self.total_steps > INITIAL_STEPS:
                        max_ego_dist = max((self.total_steps - INITIAL_STEPS) * (10.0 - physical_limit)/(FINAL_STEPS - INITIAL_STEPS) + physical_limit, 10.0)
                    ego_p = self.prng.random() * max_ego_dist + ego_lane_start
                    #print("/////+ reset: training w/physical_limit = {:.1f}, max_ego_dist = {:.1f}, ego_p = {:.1f}" #TODO debug only
                    #      .format(physical_limit, max_ego_dist, ego_p))

                # For level 4 we want to sync the agent in lane 2 to force it to avoid a collision by positioning it right next to the neighbors
                if self.difficulty_level == 4  and  ego_lane_id == 2:
                    loc = int(self.prng.random()*5.0)
                    ego_speed = self.initialize_ramp_vehicle_speed(loc, ego_p)
                else:
                    ego_speed = self.prng.random() * (SimpleHighwayRamp.MAX_SPEED - 5.0) + 5.0 #not practical to train at really low speeds

            else: #undefined levels
                ego_p = self.prng.random() * 3.0*SimpleHighwayRamp.VEHICLE_LENGTH + ego_lane_start
                ego_speed = self.prng.random() * (SimpleHighwayRamp.MAX_SPEED - 15.0) + 15.0 #not practical to train at really low speeds

        #
        #..........Initialize ego vehicle for inference
        #

        # Else, we are doing inference, so allow configrable overrides if present
        else:
            ego_lane_id = int(self.prng.random()*3) if self.init_ego_lane is None  else  self.init_ego_lane
            ego_p = self.prng.random() * 20.0*SimpleHighwayRamp.VEHICLE_LENGTH if self.init_ego_dist is None  else  self.init_ego_dist
            ego_p += self.roadway.get_lane_start_p(ego_lane_id)

            # If difficulty level 4, then always use the config value for merge relative position
            if self.difficulty_level == 4  and  ego_lane_id == 2:
                ego_speed = self.initialize_ramp_vehicle_speed(self.merge_relative_pos, ego_p)
            else:
                ego_speed = self.prng.random() * 31.0 + 5.0 if self.init_ego_speed is None  else  self.init_ego_speed

        if self.debug > 1:
            print("///// reset: ego defined: lane = {}, p = {:.1f}, speed = {:.1f}".format(ego_lane_id, ego_p, ego_speed))
        #print("///// reset: ego_lane_id = {}, ego_p = {:6.1f}, max_ego_dist = {:6.1f}, total_steps = {}"
        #      .format(ego_lane_id, ego_p, max_ego_dist, self.total_steps))

        # Reinitialize the ego vehicle and the whole observation vector
        ego_rem, lid, la, lb, l_rem, rid, ra, rb, r_rem = self.roadway.get_current_lane_geom(ego_lane_id, ego_p)
        self.vehicles[0].lane_id = ego_lane_id
        self.vehicles[0].p = ego_p
        self.vehicles[0].cur_speed = ego_speed
        self.vehicles[0].lane_change_status = "none"
        # Ego vehicle doesn't use a target speed, so it can be left untouched here.

        #
        #..........Initialize neighbor vehicles, if applicable
        #

        # Levels 0-3, ensure no neighbors participate
        if self.difficulty_level < 4:
            for n in range(1, len(self.vehicles)):
                self.vehicles[n].active = False

        # Level 4 (3 steady speed vehicles in lane 1)
        elif self.difficulty_level == 4:
            n_loc = max(self.neighbor_start_loc + self.prng.random()*6.0*SimpleHighwayRamp.VEHICLE_LENGTH, 0.0)
            n_speed = self.neighbor_speed * (self.prng.random()*0.2 + 0.9) #assigned speed +/- 10%
            if self.prng.random() > 0.9: #sometimes there will be no neighbor vehicles participating
                n_speed = 0.0
            if self.neighbor_print_latch:
                print("///// reset worker {}: Neighbor vehicles on the move in level 4. Episode {}, step {}, loc = {:.1f}, speed = {:.1f}"
                        .format(self.rollout_id, self.episode_count, self.total_steps, n_loc, n_speed))
                self.neighbor_print_latch = False

            # Initialize the 3 neighbor vehicles. In level 4 neighbor vehicles always go a constant speed, always travel in lane 1
            # (for this level, the neighbors don't use their target speed, so it doesn't need to be set here).
            n_lane_id = 1
            self.vehicles[1].lane_id = n_lane_id
            self.vehicles[1].p = n_loc + 6.0*SimpleHighwayRamp.VEHICLE_LENGTH #in front of vehicle n2
            self.vehicles[1].cur_speed = n_speed
            self.vehicles[1].lane_change_status = "none"
            self.vehicles[2].lane_id = n_lane_id
            self.vehicles[2].p = n_loc + 3.0*SimpleHighwayRamp.VEHICLE_LENGTH #in front of vehicle n3
            self.vehicles[2].cur_speed = n_speed
            self.vehicles[2].lane_change_status = "none"
            self.vehicles[3].lane_id = n_lane_id
            self.vehicles[3].p = n_loc + 0.0 #at end of the line of 3 neighbors
            self.vehicles[3].cur_speed = n_speed
            self.vehicles[3].lane_change_status = "none"

            # Mark the other neighbors as unavailable
            if SimpleHighwayRamp.NUM_NEIGHBORS > 3:
                for n in range(4, len(self.vehicles)):
                    self.vehicles[n].active = False
            if self.debug > 1:
                print("      in reset: vehicles = ")
                for i, v in enumerate(self.vehicles):
                    v.print(i)

            # If ego vehicle is in lane 1 (where neighbors start) then we need to initialize its position so that it isn't
            # going to immediately crash with them (n1 is always the farthest downtrack and n3 is always the farthest uptrack). Give it
            # more room if going in front of the neighbors, as ego has limited accel and may be starting much slower than they are.
            if ego_lane_id == 1:
                min_loc = self.vehicles[3].p - 3.0*SimpleHighwayRamp.VEHICLE_LENGTH
                max_loc = self.vehicles[1].p + 6.0*SimpleHighwayRamp.VEHICLE_LENGTH
                midway = 0.5*(max_loc - min_loc) + min_loc
                if midway < ego_p < max_loc  or  min_loc < 0.0:
                    ego_p = max_loc
                    if self.debug > 0:
                        print("///// reset adjusting agent to: lane = {}, speed = {:.2f}, p = {:.2f}".format(ego_lane_id, ego_speed, ego_p))
                elif min_loc < ego_p <= midway:
                    ego_p = min_loc
                    if self.debug > 0:
                        print("///// reset adjusting agent to: lane = {}, speed = {:.2f}, p = {:.2f}".format(ego_lane_id, ego_speed, ego_p))

        # Level 5 (neighbors have differing speeds, lanes and starting locations, and have ACC capability to avoid forward collisions)
        elif self.difficulty_level == 5:

            # Place the neighbors, one at a time
            for n in range(1, len(self.vehicles)): #don't look at the ego vehicle (0)

                lane_id = 0
                loc = 0.0

                # For neighbor #1, frequently force it to be downtrack of the ego vehicle, and #2 similarly downtrack but in the next
                # through lane, both to force learning of a rear-end situation
                if n <= 2  and  self.prng.random() < 0.9:
                    if n == 1:
                        lane_id = ego_lane_id
                    elif ego_lane_id == 1:
                        lane_id = 0
                    else:
                        lane_id = 1
                    loc = ego_p + min(self.prng.random()*200.0 + 3.0*SimpleHighwayRamp.VEHICLE_LENGTH, self.roadway.get_total_lane_length(lane_id))

                # Else, choose the neighbor's location, but avoid being too close to another vehicle, longitudinally
                else:
                    space_found = False
                    while not space_found:
                        lane_id = 0
                        draw = self.prng.random()
                        if draw < 0.333:
                            lane_id = 1
                        elif draw < 0.667:
                            lane_id = 2
                        lane_begin = self.roadway.get_lane_start_p(lane_id)
                        max_dist = min(max(max_ego_dist, 0.3*SimpleHighwayRamp.SCENARIO_LENGTH), self.roadway.get_total_lane_length(lane_id) - 50.0)
                        loc = self.prng.random()*max_dist + lane_begin
                        space_found = self._verify_safe_location(n, lane_id, loc)

                # Store the initial state of this neighbor
                self.vehicles[n].lane_id = lane_id
                self.vehicles[n].p = loc
                min_speed = 0.7*SimpleHighwayRamp.ROAD_SPEED_LIMIT
                max_speed = SimpleHighwayRamp.MAX_SPEED
                speed = self.prng.random()*(max_speed - min_speed) + min_speed
                self.vehicles[n].cur_speed = speed
                self.vehicles[n].tgt_speed = speed #this will be the speed the ACC controller always tries to hit
                self.vehicles[n].lane_change_status = "none"
                #print("/////+ reset: neighbor {} lane = {}, p = {:6.1f}, speed = {:4.1f}".format(n, lane_id, loc, self.vehicles[n].cur_speed))

        # Undefined levels
        else:
            raise NotImplementedError("///// Neighbor vehicle motion not defined for difficulty level {}".format(self.difficulty_level))

        #print("///// reset: training = {}, ego_lane_id = {}, ego_p = {:.2f}, ego_speed = {:.2f}".format(self.training, ego_lane_id, ego_p, ego_speed))
        self._verify_obs_limits("reset after initializing local vars")

        #
        #..........Wrap up
        #

        # Reinitialize the remainder of the observation vector
        self.obs = np.zeros(self.OBS_SIZE, dtype = float)
        self.obs[self.LC_CMD]               = LaneChange.STAY_IN_LANE
        self.obs[self.LC_CMD_PREV]          = LaneChange.STAY_IN_LANE
        self.obs[self.EGO_SPEED]            = ego_speed
        self.obs[self.EGO_SPEED_PREV]       = ego_speed
        self.obs[self.EGO_DES_ACCEL]        = 0.0
        self.obs[self.EGO_DES_ACCEL_PREV]   = 0.0
        self.obs[self.EGO_LANE_REM]         = ego_rem
        self.obs[self.STEPS_SINCE_LN_CHG]   = SimpleHighwayRamp.MAX_STEPS_SINCE_LC
        self._verify_obs_limits("reset after populating main obs with ego stuff")
        self._update_obs_zones()

        # Other persistent data
        self.lane_change_count = 0
        self.steps_since_reset = 0
        self.stopped_count = 0
        self.reward_for_completion = True
        self.episode_count += 1
        self._verify_obs_limits("end of reset")

        if self.debug > 0:
            print("///// End of reset().")
        return self.obs, {}


    def step(self,
                cmd     : list      #list of floats; 0 = speed command, 1 = desired lane, scaled
            ) -> Tuple[np.array, float, bool, bool, Dict]:

        """Wrapper around the real step method to trap unhandled exceptions."""

        try:
            return self._step(cmd)
        except Exception as e:
            print("\n///// Exception trapped in SimpleHighwayRamp.step: ", e)


    def _step(self,
                cmd     : list      #list of floats; 0 = speed command, 1 = desired lane, scaled
            ) -> Tuple[np.array, float, bool, bool, Dict]:

        """Executes a single time step of the environment.  Determines how the input commands (actions) will alter the
            simulated world and returns the resulting observations to the agent.

            Return is array of new observations, new reward, done flag, truncated flag, and a dict of additional info.

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!
        """

        if self.debug > 0:
            print("\n///// Entering step(): cmd = ", cmd)
            print("      vehicles array contains:")
            for i, v in enumerate(self.vehicles):
                v.print(i)

        self.total_steps += 1
        self.steps_since_reset += 1
        done = False
        return_info = {"reason": "Unknown"}

        #
        #..........Update longitudinal state for all vehicles
        #

        # Apply command masking for first few steps to avoid startup problems with the feedback observations
        action = [None]*2
        action[0] = cmd[0]
        action[1] = cmd[1]
        if self.steps_since_reset < 1:
            action[1] = 0.0

        # Unscale the action inputs (both actions are in [-1, 1])
        desired_accel = action[0] * SimpleHighwayRamp.MAX_ACCEL
        lc_cmd = int(math.floor(action[1] + 0.5))
        #print("///// step: incoming cmd[1] = {:5.2f}, lc_cmd = {:2}, current lane = {}, p = {:7.2f}, steps = {}"
        #      .format(cmd[1], lc_cmd, self.vehicles[0].lane_id, self.vehicles[0].p, self.steps_since_reset))

        # Move the ego vehicle downtrack. This doesn't account for possible lane changes, which are handled seperately, below.
        new_ego_speed, new_ego_p = self.vehicles[0].advance_vehicle_accel(desired_accel)
        if new_ego_p > SimpleHighwayRamp.SCENARIO_LENGTH:
            new_ego_p = SimpleHighwayRamp.SCENARIO_LENGTH #limit it to avoid exceeding NN input validation rules
        if self.debug > 1:
            print("      Vehicle 0 advanced with desired_accel = {:.2f}. new_speed = {:.2f}, new_p = {:.2f}"
                    .format(desired_accel, new_ego_speed, new_ego_p))

        # Move each of the active neighbor vehicles downtrack.
        for n in range(1, len(self.vehicles)):
            if not self.vehicles[n].active:
                continue
            new_speed_cmd = self.vehicles[n].cur_speed
            if self.difficulty_level == 5:
                new_speed_cmd = self._acc_speed_control(n)
            new_speed, new_p = self.vehicles[n].advance_vehicle_spd(new_speed_cmd)

            # Since neighbor vehicles may not change lanes, we need to take them out of action if they run off the end.
            lane = self.vehicles[n].lane_id
            lane_end = self.roadway.get_lane_start_p(lane) + self.roadway.get_total_lane_length(lane)
            if new_p > lane_end:
                self.vehicles[n].active = False
            if self.debug > 1:
                print("      Neighbor {} (lane {}) advanced with new_speed_cmd = {:.2f}. new_speed = {:.2f}, new_p = {:.2f}"
                        .format(n, self.vehicles[n].lane_id, new_speed_cmd, new_speed, new_p))

        # Update ego vehicle obs vector
        self.obs[self.EGO_SPEED_PREV] = self.obs[self.EGO_SPEED]
        self.obs[self.EGO_SPEED] = new_ego_speed
        self.obs[self.EGO_DES_ACCEL_PREV] = self.obs[self.EGO_DES_ACCEL]
        self.obs[self.EGO_DES_ACCEL] = desired_accel
        if new_ego_p >= SimpleHighwayRamp.SCENARIO_LENGTH:
            done = True
            return_info["reason"] = "SUCCESS - end of scenario!"
            #print("/////+ step: {} step {}, success - completed the track".format(self.rollout_id, self.total_steps))  #TODO debug
        self._verify_obs_limits("step after moving vehicles forward")

        #
        #..........Update lane change status for ego vehicle
        #

        # Determine if we are beginning or continuing a lane change maneuver.
        # Accept a lane change command that lasts for several time steps or only one time step.  Once the first
        # command is received (when currently not in a lane change), then start the maneuver and ignore future
        # lane change commands until the underway maneuver is complete, which takes several time steps.
        # It's legal, but not desirable, to command opposite lane change directions in consecutive time steps.
        ran_off_road = False
        if lc_cmd != LaneChange.STAY_IN_LANE  or  self.vehicles[0].lane_change_status != "none":
            if self.vehicles[0].lane_change_status == "none": #count should always be 0 in this case, so initiate a new count
                if lc_cmd == LaneChange.CHANGE_LEFT:
                    self.vehicles[0].lane_change_status = "left"
                else:
                    self.vehicles[0].lane_change_status = "right"
                self.lane_change_count = 1
                if self.debug > 0:
                    print("      *** New lane change maneuver initiated. lc_cmd = {}, status = {}"
                            .format(lc_cmd, self.vehicles[0].lane_change_status))
            else: #once a lane change is underway, continue until complete, regardless of new commands
                self.lane_change_count += 1

        # Check that an adjoining lane is available in the direction commanded until maneuver is complete
        new_ego_lane = int(self.vehicles[0].lane_id)
        tgt_lane = new_ego_lane
        if self.lane_change_count > 0:

            # If we are still in the original lane then
            if self.lane_change_count <= SimpleHighwayRamp.HALF_LANE_CHANGE_STEPS:
                # Ensure that there is a lane to change into and get its ID
                tgt_lane = self.roadway.get_target_lane(int(self.vehicles[0].lane_id), self.vehicles[0].lane_change_status, new_ego_p)
                if tgt_lane < 0:
                    done = True
                    ran_off_road = True
                    return_info["reason"] = "Ran off road; illegal lane change"
                    if self.debug > 1:
                        print("      DONE!  illegal lane change commanded.")
                    #print("/////+ step: {} step {}, illegal lane change".format(self.rollout_id, self.total_steps))  #TODO debug

                # Else, we are still going; if we are exactly half-way then change the current lane ID
                elif self.lane_change_count == SimpleHighwayRamp.HALF_LANE_CHANGE_STEPS:
                    new_ego_lane = tgt_lane

            # Else, we have already crossed the dividing line and are now mostly in the target lane
            else:
                coming_from = "left"
                if self.vehicles[0].lane_change_status == "left":
                    coming_from = "right"
                # Ensure the lane we were coming from is still adjoining (since we still have 2 wheels there)
                prev_lane = self.roadway.get_target_lane(tgt_lane, coming_from, new_ego_p)
                if prev_lane < 0: #the lane we're coming from ended before the lane change maneuver completed
                    done = True
                    ran_off_road = True
                    return_info["reason"] = "Ran off road; lane change initiated too late"
                    if self.debug > 1:
                        print("      DONE!  original lane ended before lane change completed.")
                    #print("/////+ step: {} step {}, late lane change".format(self.rollout_id, self.total_steps))  #TODO debug

        #
        #..........Manage lane change for any neighbors in lane 2
        #

        # Loop through all active neighbors, looking for any that are in lane 2
        for n in range(1, len(self.vehicles)):
            v = self.vehicles[n]
            if not v.active:
                continue

            if v.lane_id == 2:

                # If it is in the merge zone, then
                progress = v.p - self.roadway.get_lane_start_p(2)
                l2_length = self.roadway.get_total_lane_length(2)
                if progress > 0.7*l2_length:

                    # Randomly decide if it's time to do a lane change
                    if self.prng.random() < 0.05  or  l2_length - progress < 150.0:

                        # Look for a vehicle beside it in lane 1
                        safe = True
                        for j in range(len(self.vehicles)):
                            if j == n:
                                continue
                            if self.vehicles[j].lane_id == 1  and  abs(self.vehicles[j].p - v.p) < 2.0*SimpleHighwayRamp.VEHICLE_LENGTH:
                                safe = False
                                break

                        # If it is safe to move, then just do an immediate lane reassignment (no multi-step process like ego does)
                        if safe:
                            v.lane_id = 1

                        # Else it is being blocked, then slow down a bit
                        else:
                            v.cur_speed *= 0.8

        #
        #..........Update ego vehicle's understanding of roadway geometry and various termination conditions
        #

        # Get updated metrics of ego vehicle relative to the new lane geometry
        new_ego_rem, lid, la, lb, l_rem, rid, ra, rb, r_rem = self.roadway.get_current_lane_geom(new_ego_lane, new_ego_p)

        #TODO - for debugging only, this whole section:
        if not self.training:
            if self.lane_change_count == SimpleHighwayRamp.HALF_LANE_CHANGE_STEPS - 1:
                print("   ** LC next step: ego_p = {:.1f}, ego_rem = {:.1f}, lid = {}, la = {:.1f}, lb = {:.1f}, l_rem = {:.1f}".format(new_ego_p, new_ego_rem, lid, la, lb, l_rem))
            elif self.lane_change_count == SimpleHighwayRamp.HALF_LANE_CHANGE_STEPS:
                print("   ** LC now: ego_p = {:.1f}, ego_rem = {:.1f}, rid = {}, ra = {:.1f}, rb = {:.1f}, r_rem = {:.1f}".format(new_ego_p, new_ego_rem, rid, ra, rb, r_rem))

        # If remaining lane distance has gone away, then vehicle has run straight off the end of the lane, so episode is done
        if new_ego_rem <= 0.0:
            new_ego_rem = 0.0 #clip it so that obs space isn't violated
            if not done:
                done = True
                ran_off_road = True
                return_info["reason"] = "Ran off end of terminating lane"
                #print("/////+ step: {} step {}, off end of terminating lane".format(self.rollout_id, self.total_steps))  #TODO debug

        # Update counter for time in between lane changes
        if self.obs[self.STEPS_SINCE_LN_CHG] < SimpleHighwayRamp.MAX_STEPS_SINCE_LC:
            self.obs[self.STEPS_SINCE_LN_CHG] += 1

        # If current lane change is complete, then reset its state and counter
        if self.lane_change_count >= SimpleHighwayRamp.TOTAL_LANE_CHANGE_STEPS:
            self.vehicles[0].lane_change_status = "none"
            self.lane_change_count = 0
            self.obs[self.STEPS_SINCE_LN_CHG] = SimpleHighwayRamp.TOTAL_LANE_CHANGE_STEPS

        self.vehicles[0].lane_id = new_ego_lane
        if self.debug > 0:
            print("      step: done lane change. underway = {}, new_ego_lane = {}, tgt_lane = {}, count = {}, done = {}, steps since = {}"
                    .format(self.vehicles[0].lane_change_status, new_ego_lane, tgt_lane, self.lane_change_count, done, self.obs[self.STEPS_SINCE_LN_CHG]))

        # Update the obs vector with the new state info
        self.obs[self.EGO_LANE_REM] = new_ego_rem
        self.obs[self.LC_CMD_PREV] = self.obs[self.LC_CMD]
        self.obs[self.LC_CMD] = lc_cmd
        self._update_obs_zones()
        self._verify_obs_limits("step after updating obs vector")

        # If vehicle has been stopped for several time steps, then declare the episode done as a failure
        stopped_vehicle = False
        if self.vehicles[0].cur_speed < 0.5:
            self.stopped_count += 1
            if self.stopped_count > 3:
                done = True
                stopped_vehicle = True
                return_info["reason"] = "Vehicle is crawling to a stop"
                #print("/////+ step: {} step {}, vehicle stopped".format(self.rollout_id, self.total_steps))  #TODO debug
        else:
            self.stopped_count = 0

        # Check that none of the vehicles has crashed into another, accounting for a lane change in progress
        # taking up both lanes. Do this check last, as it is the most severe failure, and needs to override
        # the others in the reward evaluation.
        crash = self._check_for_collisions()
        if crash:
            done = True
            return_info["reason"] = "Crashed into neighbor vehicle"
            #print("/////+ step: {} step {}, crash!".format(self.rollout_id, self.total_steps))  #TODO debug

        # Determine the reward resulting from this time step's action
        reward, expl = self._get_reward(done, crash, ran_off_road, stopped_vehicle)
        return_info["reward_detail"] = expl
        #print("/////+ step: {} step {}, returning reward of {}, {}".format(self.rollout_id, self.total_steps, reward, expl))  #TODO debug

        # Verify that the obs are within design limits
        self._verify_obs_limits("step after reward calc")

        if self.debug > 0:
            print("///// step complete. Returning obs = ")
            print(      self.obs)
            print("      reward = ", reward, ", done = ", done)
            print("      final vehicles array =")
            for i, v in enumerate(self.vehicles):
                v.print(i)
            print("      reason = {}".format(return_info["reason"]))
            print("      reward_detail = {}\n".format(return_info["reward_detail"]))

        truncated = False #indicates if the episode ended prematurely due to step/time limit
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

        assert 0 <= vehicle_id < len(self.vehicles), \
                "///// SimpleHighwayRamp.get_vehicle_dist_downtrack: illegal vehicle_id entered: {}".format(vehicle_id)

        ddt = self.vehicles[vehicle_id].p
        lane_id = self.vehicles[vehicle_id].lane_id
        if lane_id == 2:
            ddt -= self.roadway.get_lane_start_p(lane_id)
        return ddt


    def get_vehicle_data(self) -> List:
        """Returns a list of all the vehicles in the scenario, with the ego vehicle as the first item."""

        return self.vehicles


    def close(self):
        """Closes any resources that may have been used during the simulation."""
        pass #this method not needed for this version


    def initialize_ramp_vehicle_speed(self,
                                      relative_pos    : int = 2,  #desired position of the ego vehicle relative to the 3 neighbors
                                                                  # at the time it reaches the merge area
                                      ego_p           : float = 0.0 #ego vehicle's starting P location, m
                                     ) -> float:
        """Returns a speed to start the ego vehicle when it starts at the beginning of lane 2, such that when it arrives at the
            beginning of the merge area, it will approximately match the specified position relative to the approaching neighbor vehicles.
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
        ns = self.neighbor_speed if self.neighbor_speed > 0.0  else  SimpleHighwayRamp.ROAD_SPEED_LIMIT
        if relative_pos <= 1:
            tgt_arrival_time = (L1_DIST_TO_MERGE - self.neighbor_start_loc - 2*headway) / ns
        elif relative_pos == 2:
            tgt_arrival_time = (L1_DIST_TO_MERGE - self.neighbor_start_loc - headway) / ns
        else:
            tgt_arrival_time = (L1_DIST_TO_MERGE - self.neighbor_start_loc) / ns
        #print("\n///// initialize_ramp_vehicle_speed: rel pos = {}, headway = {:.1f}, n start loc = {:.1f}, n spd = {:.1f}, tgt time = {:.1f}"
        #      .format(relative_pos, headway, self.neighbor_start_loc, ns, tgt_arrival_time))

        # Get a random offset from that arrival time and apply it
        time_headway = headway / ns
        offset = 0.0 #self.prng.normal(scale = 0.1*time_headway)
        if relative_pos == 0: #in front of first neighbor
            offset -= 1.1 * time_headway
        elif relative_pos == 4: #behind last neighbor
            offset += 1.1 * time_headway
        tgt_arrival_time += offset

        # If our desired arrival time is large enough then
        v0 = SimpleHighwayRamp.ROAD_SPEED_LIMIT
        #print("///// initialize_ramp_vehicle_speed: tgt time = {:.1f}, offset = {:.1f}, ego_p = {:.1f}".format(tgt_arrival_time, offset, ego_p))
        dist_to_merge = self.roadway.get_lane_start_p(2) + L2_DIST_TO_MERGE - ego_p
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

        self.ignore_neighbor_crashes = False
        try:
            inc = config["ignore_neighbor_crashes"]
            if inc:
                self.ignore_neighbor_crashes = True
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

        # Level 3 needs to emphasize lanes 0 & 2 since the agent naturally prefers lane 1
        elif self.difficulty_level == 3:

            draw = self.prng.random()
            if draw < 0.3:
                return 1
            else:
                return 2

        else:
            return int(self.prng.random()*3)


    def _update_obs_zones(self):
        """Updates the observation vector data for each of the roadway zones, based on ego state and current neighbor vehicle states.

            CAUTION: requires that each zone is represented the same in the obs vector and all zones are contiguous there.
        """

        # Determine offsets in the obs vector for zone columns and rows
        base = self.Z1_DRIVEABLE
        num_zone_fields = self.Z2_DRIVEABLE - base

        # Clear all zone info from previous time step
        for z in range(9):
            self.obs[base + num_zone_fields*z + 0] = 0.0 #drivable
            self.obs[base + num_zone_fields*z + 1] = 0.0 #reachable
            self.obs[base + num_zone_fields*z + 2] = 0.0 #occupied
            self.obs[base + num_zone_fields*z + 3] = 0.0 #p
            self.obs[base + num_zone_fields*z + 4] = 0.0 #speed
        self.obs[self.NEIGHBOR_IN_EGO_ZONE] = 0.0

       # Get the current roadway geometry
        ego_lane_id = self.vehicles[0].lane_id
        ego_p = self.vehicles[0].p
        if self.debug > 1:
            print("///// Entering update_obs_zones: ego_lane_id = {}, ego_p = {:.1f}, base = {}"
                  .format(ego_lane_id, ego_p, base))

        # Determine pavement existence and reachability in each zone
        # CAUTION: this block is dependent on the specific roadway geometry for this experiment, and is not generalized
        for row in range(1, 4):
            zone_front = ((3 - row) + 0.5)*SimpleHighwayRamp.OBS_ZONE_LENGTH #distance downtrack from ego vehicle, m
            zone_rear = zone_front - SimpleHighwayRamp.OBS_ZONE_LENGTH
            zone_mid_p = ego_p + 0.5*(zone_front + zone_rear) #absolute coordinate in p-frame
            # Get updated roadway geometry; NB all distances returned are relative to current ego location
            ego_rem, lid, la, lb, l_rem, rid, ra, rb, r_rem = self.roadway.get_current_lane_geom(ego_lane_id, ego_p)
            if self.debug > 1:
                print("///// _update_obs_zones: row = {}, ego_p = {:.1f}, zone_front = {:.1f}, zone_rear = {:.1f}, zone_mid = {:.1f}, la = {:.1f}, lb = {:.1f}"
                        .format(row, ego_p, zone_front, zone_rear, zone_mid_p, la, lb))

            # Determine if there is pavement in the left-hand zone and it's reachable
            if lid >= 0: #the lane exists somewhere along the route

                # Determine if the left lane exists next to the middle of this zone
                start_p = self.roadway.get_lane_start_p(lid)
                if start_p <= zone_mid_p <= start_p + self.roadway.get_total_lane_length(lid):
                    l_zone = 3*(row - 1) + 1
                    l_offset = base + (l_zone - 1)*num_zone_fields
                    self.obs[l_offset + 0] = 1.0 #drivable
                    if la <= zone_front  and  lb >= zone_rear:
                        self.obs[l_offset + 1] = 1.0 #reachable

            # Determine if there is pavement in the right-hand zone and it's reachable
            if rid >= 0: #there's a lane to the right somewhere along this route

                # Determine if the right lane exists next to the middle of this zone
                start_p = self.roadway.get_lane_start_p(rid)
                if start_p <= zone_mid_p <= start_p + self.roadway.get_total_lane_length(rid):
                    r_zone = 3*(row - 1) + 3
                    r_offset = base + (r_zone - 1)*num_zone_fields
                    self.obs[r_offset + 0] = 1.0 #drivable
                    if ra <= zone_front  and  rb >= zone_rear:
                        self.obs[r_offset + 1] = 1.0 #reachable

        # We know there's a lane in the center, but not how far it extends in either direction so look at each zone in this column.
        # Note that the "reachable" determination here is different from those above. Since the above represent adjacent lanes, they
        # can afford to be more liberal in that they say reachable = True if any part of the adjacent pavement borders any part of the
        # zone in question (i.e. the entire zone edge does not need to touch adjacent pavement). Whereas, for the ego's own lane, it
        # is more important to know as soon as the pavement disappears in _any_ part of its forward zone, so it can prepare to change
        # lanes.
        for row in range(1, 5):
            zone = 3*(row - 1) + 2
            if row == 3: #ego zone
                continue
            elif row == 4:
                zone = 8
            offset = base + (zone - 1)*num_zone_fields
            zone_front = ((3 - row) + 0.5)*SimpleHighwayRamp.OBS_ZONE_LENGTH #distance downtrack from ego vehicle, m
            zone_rear = zone_front - SimpleHighwayRamp.OBS_ZONE_LENGTH
            if ego_rem >= zone_front: #don't worry about lane existence behind ego vehicle; assume it's there
                self.obs[offset + 0] = 1.0 #drivable
                self.obs[offset + 1] = 1.0 #reachable is guaranteed if it is driveable, since it's the same lane

        # Loop through the neighbor vehicles
        for neighbor_idx in range(1, len(self.vehicles)):
            nv = self.vehicles[neighbor_idx]

            # Find which zone column it is in (relative lane), if any (could be 2 lanes away) (ego is in column 1, lanes are 0-indexed, left-to-right)
            column = nv.lane_id - ego_lane_id + 1
            if self.debug > 1:
                print("///// update_obs_zones: considering neighbor {} in column {}".format(neighbor_idx, column))

            # Find which zone row it is in, if any (could be too far away)
            row = 0
            dist_ahead_of_ego = nv.p - ego_p
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
                if self.debug > 1:
                    print("///// update_obs_zones: found neighbor {} too far away, with column {}, row {}, p = {:.1f}, dist_ahead_of_ego = {:.1f}"
                          .format(neighbor_idx, column, row, nv.p, dist_ahead_of_ego))
                continue

            # Neighbor is within our obs zone grid.
            # If it is also within the ego zone then
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
                    if column == 1:
                        zone = 8
                    else:
                        if self.debug > 1:
                            print("///// update_obs_zones: found a neighbor beside zone 8: column = {}, dist_ahead_of_ego = {:.1f}"
                                  .format(column, dist_ahead_of_ego))
                        continue

                offset = base + (zone - 1)*num_zone_fields
                if self.debug > 1:
                    print("///// update_obs_zones: neighbor offset = {} for zone {}".format(offset, zone))

                # Since we've identified a neighbor vehicle in this zone, flag it as occupied
                self.obs[offset + 2] = 1.0

                # Set the neighbor's relative location within the zone
                zone_rear_p = ego_p + ((2.0 - row) + 0.5)*SimpleHighwayRamp.OBS_ZONE_LENGTH
                rel_p = (nv.p - zone_rear_p) / SimpleHighwayRamp.OBS_ZONE_LENGTH
                self.obs[offset + 3] = rel_p

                # Set the neighbor's relative speed
                self.obs[offset + 4] = (nv.cur_speed - self.vehicles[0].cur_speed) / SimpleHighwayRamp.ROAD_SPEED_LIMIT

                if self.debug > 1:
                    print("///// update_obs_zones: neighbor {} has column = {}, row = {}, zone = {}, zone_rear = {:.1f}, rel_p = {:.2f}, ego speed = {:.1f}"
                          .format(neighbor_idx, column, row, zone, zone_rear_p, rel_p, self.vehicles[0].cur_speed))

        if self.debug > 1:
            print("///// update_obs_zones complete.")
            for zone in range(1, 10):
                offset = base + (zone - 1)*num_zone_fields
                print("      Zone {}: drivable = {:.1f}, reachable = {:.1f}, occupied = {:.1f}, rel p = {:.2f}, rel speed = {:.2f}"
                      .format(zone, self.obs[offset+0], self.obs[offset+1], self.obs[offset+2], self.obs[offset+3], self.obs[offset+4]))


    def _verify_safe_location(self,
                              n         : int,  #neighbor ID
                              lane_id   : int,  #desired lane ID for the neighbor
                              p         : float,#desired P coordinate for the neighbor (m in paremetric frame)
                             ) -> bool:         #returns true if the indicated location is safe
        """Determines if the candidate location (lane & P coordinate) is a safe place to put a vehicle at the beginning of a scenario.
            It needs to be sufficiently far from any other neighbors whose starting locations have already been defined.
        """

        assert 0 <= lane_id < SimpleHighwayRamp.NUM_LANES, "///// Attempting to place neighbor {} in invalid lane {}".format(n, lane_id)
        start = self.roadway.get_lane_start_p(lane_id)
        assert start <= p < start + self.roadway.get_total_lane_length(lane_id), \
                "///// Attempting to place neighbor {} in lane {} at invalid p = {:.1f}".format(n, lane_id, p)

        safe = True

        # Loop through all active vehicles
        for o in range(len(self.vehicles)):
            other = self.vehicles[o]
            if not other.active:
                continue

            # If the other vehicle is in candiate's lane then check if it is too close longitudinally. Note that if a neighbor has
            # not yet been placed, its lane ID is -1
            if other.lane_id == lane_id:
                if 0.0 <= p - other.p < 5.0*SimpleHighwayRamp.VEHICLE_LENGTH  or \
                   0.0 <= other.p - p < 3.0*SimpleHighwayRamp.VEHICLE_LENGTH:
                    safe = False

        return safe


    def _acc_speed_control(self,
                           n        : int   #ID of the neighbor in question; ASSUMED to be > 0
                          ) -> float:       #returns speed command, m/s
        """Applies a crude adaptive cruise control logic to the specified neighbor vehicle so that it attempts to follow it's target speed
            whenever possible, but slows to match the speed of a slower vehicle close in front of it to avoid a crash.
        """

        speed_cmd = self.vehicles[n].tgt_speed

        # Loop through all other active vehicles in the scenario
        for i in range(len(self.vehicles)): #includes ego vehicle as #0
            if i != n  and  self.vehicles[i].active:

                # If that vehicle is close in front of us then
                if self.vehicles[i].lane_id == self.vehicles[n].lane_id:
                    dist = self.vehicles[i].p - self.vehicles[n].p
                    if 0.0 < dist <= SimpleHighwayRamp.DISTANCE_OF_CONCERN:

                        # Reduce our speed command gradually toward that vehicle's speed, to avoid a collision. Since there could be multiple
                        # vehicles within the distance of concern, the limiter must account for the results of a previous iteration of this loop.
                        fwd_speed = self.vehicles[i].cur_speed #speed of the forward vehicle
                        if fwd_speed < self.vehicles[n].cur_speed:
                            f = (dist - SimpleHighwayRamp.CRITICAL_DISTANCE) / \
                                (SimpleHighwayRamp.DISTANCE_OF_CONCERN - SimpleHighwayRamp.CRITICAL_DISTANCE)
                            speed_cmd = min(max(f*(self.vehicles[n].tgt_speed - fwd_speed) + fwd_speed, fwd_speed), speed_cmd)
                            #print("///// ** Neighbor {} ACC is active!  tgt_speed = {:.1f}, speed_cmd = {:.1f}, dist = {:5.1f}, fwd_speed = {:.1f}"
                            #    .format(n, self.vehicles[n].tgt_speed, speed_cmd, dist, fwd_speed))

        return speed_cmd


    def _check_for_collisions(self) -> bool:
        """Compares location and bounding box of each vehicle with all other vehicles to determine if there are
            any overlaps.  If any two vehicle bounding boxes overlap, then returns True, otherwise False.

            Return: has there been a collision we are interested in?
            Note that if the collision is between two neighbors (ego not involved) then return value depends on
            the config setting "ignore_neighbor_crashes".
        """

        if self.debug > 1:
            print("///// Entering _check_for_collisions")
        crash = False

        # Loop through all active vehicles but the final one to get vehicle A
        for i in range(len(self.vehicles) - 1):
            va = self.vehicles[i]
            if not va.active:
                continue

            # Loop through the remaining active vehicles to get vehicle B
            for j in range(i + 1, len(self.vehicles)):
                vb = self.vehicles[j]
                if not vb.active:
                    continue

                # If A and B are in the same lane, then
                if va.lane_id == vb.lane_id:

                    # If they are within one car length of each other, it's a crash
                    if abs(va.p - vb.p) <= SimpleHighwayRamp.VEHICLE_LENGTH:

                        # Mark the involved vehicles as out of service
                        va.active = False
                        vb.active = False
                        va.crashed = True
                        vb.crashed = True

                        # Mark it so only if it involves the ego vehicle or we are worried about all crashes
                        if i == 0  or  j == 0  or  not self.ignore_neighbor_crashes:
                            crash = True
                            if self.debug > 1:
                                print("      CRASH in same lane between vehicles {} and {} near {:.2f} m in lane {}"
                                        .format(i, j, va.p, va.lane_id))
                            break

                # Else if they are in adjacent lanes, then
                elif abs(va.lane_id - vb.lane_id) == 1:

                    # If either vehicle is changing lanes at the moment, then
                    if va.lane_change_status != "none"  or  vb.lane_change_status != "none":

                        # If the lane changer's target lane is occupied by the other vehicle, then
                        va_tgt = self.roadway.get_target_lane(va.lane_id, va.lane_change_status, va.p)
                        vb_tgt = self.roadway.get_target_lane(vb.lane_id, vb.lane_change_status, vb.p)
                        if va_tgt == vb.lane_id  or  vb_tgt == va.lane_id:

                            # If the two are within a vehicle length of each other, then it's a crash
                            if abs(va.p - vb.p) <= SimpleHighwayRamp.VEHICLE_LENGTH:

                                # Mark the involved vehicles as out of service
                                va.active = False
                                vb.active = False
                                va.crashed = True
                                vb.crashed = True

                                # Mark it so only if it involves the ego vehicle or we are worried about all crashes
                                if i == 0  or  j == 0  or  not self.ignore_neighbor_crashes:
                                    crash = True
                                    if self.debug > 1:
                                        print("      CRASH in adjacent lanes between vehicles {} and {} near {:.2f} m in lane {}"
                                                .format(i, j, vb.p, va.lane_id))
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
            print("///// Entering _get_reward rollout {}, step {}. done = {}, crash = {}, off_road = {}"
                    .format(self.rollout_id, self.total_steps, done, crash, off_road))
        reward = 0.0
        explanation = ""

        # If the episode is done then
        if done:

            # If there was a multi-car crash or off-roading (single-car crash) then set a penalty, larger for multi-car crash
            if crash:
                reward = -15.0
                explanation = "Crashed into a vehicle. "

            elif off_road:
                reward = -10.0
                # Assign a different value if agent is in lane 2 so we can see in the logs which episodes are in this lane
                if self.vehicles[0].lane_id == 2:
                    reward = -10.0
                explanation = "Ran off road. "

            # Else if the vehicle just stopped in the middle of the road then
            elif stopped:

                # Subtract a penalty for no movement (needs to be as severe as off-road)
                reward = -12.0
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
                        dist = abs(self.steps_since_reset - 130)
                        reward = min(max(10.0 - 0.2*dist, 0.0), 10.0)
                        explanation = "Successful episode! {} steps".format(self.steps_since_reset)
                    else:
                        explanation = "Completed episode, but no bonus due to rule violation."

        # Else, episode still underway
        else:

            # Reward for staying alive
            INITIAL_BONUS = 0.01
            bonus = INITIAL_BONUS
            tune_steps = 1
            reward += bonus

            # Small penalty for widely varying lane commands
            cmd_diff = abs(self.obs[self.LC_CMD] - self.obs[self.LC_CMD_PREV])
            penalty = 0.1 * cmd_diff * cmd_diff
            reward -= penalty
            if penalty > 0.0001:
                explanation += "Ln cmd pen {:.4f}. ".format(penalty)

            # Small penalty for widely varying accel commands
            if self.difficulty_level > 0:
                cmd_diff = abs(self.obs[self.EGO_DES_ACCEL] - self.obs[self.EGO_DES_ACCEL_PREV]) / SimpleHighwayRamp.MAX_ACCEL
                penalty = 0.4 * cmd_diff * cmd_diff
                reward -= penalty
                if penalty > 0.0001:
                    explanation += "Accel cmd pen {:.4f}. ".format(penalty)

            # Penalty for deviating from roadway speed limit
            speed_mult = 0.3
            if self.difficulty_level == 1  or  self.difficulty_level == 2:
                speed_mult *= 2.0

            norm_speed = self.obs[self.EGO_SPEED] / SimpleHighwayRamp.ROAD_SPEED_LIMIT #1.0 = speed limit
            diff = abs(norm_speed - 1.0)
            penalty = 0.0
            if diff > 0.02:
                penalty = speed_mult*(diff - 0.02)
                explanation += "spd pen {:.4f}. ".format(penalty)
            reward -= penalty

            # If a lane change was initiated, apply a penalty depending on how soon after the previous lane change
            if self.lane_change_count == 1:
                penalty = 0.05 + 0.005*(SimpleHighwayRamp.MAX_STEPS_SINCE_LC - self.obs[self.STEPS_SINCE_LN_CHG])
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
            for i in range(self.OBS_SIZE):
                assert lo[i] <= self.obs[i] <= hi[i], "\n///// obs[{}] value ({}) is outside bounds {} and {}" \
                                                        .format(i, self.obs[i], lo[i], hi[i])

        except AssertionError as e:
            print(e)
            print("///// Full obs vector content at: {}:".format(tag))
            for j in range(self.OBS_SIZE):
                print("      {:2d}: {}".format(j, self.obs[j]))


######################################################################################################
######################################################################################################

"""Dummy function that is only here for backwards compatibility to checkpoints created prior to 8/10/23."""

#TODO: if we are no longer using old checkpoints, remove this.

def curriculum_fn(train_results:        dict,           #current status of training progress
                  task_settable_env:    TaskSettableEnv,#the env model that difficulties will be applied to
                  env_ctx,                              #???
                 ) -> int:                              #returns the new difficulty level

    return task_settable_env.get_task()


######################################################################################################
######################################################################################################



class Roadway:
    """Defines the geometry of the roadway lanes and their drivable connections.  All dimensions are
        physical quantities, measured in meters from an arbitrary map origin and are in the map
        coordinate frame.  The roadway being modeled looks roughly like the diagram at the top of this
        code file.  However, this class provides convertor methods to/from the parametric coordinate
        frame, which abstracts it slightly to be more of a structural "schematic" for better
        representation in our NN observation space. To that end, all lanes in the parametric frame are
        considered parallel and physically next to each other (even though the boundary separating two
        lanes may not be permeable, e.g. a jersey wall).

        All lanes go from left to right, with travel direction being to the right. The coordinate system
        is oriented so that the origin is at the left (beginning of the first lane), with the X axis
        going to the right and Y axis going upward on the page. Not all lanes have to begin at X = 0,
        but at least one does. Others may begin at X > 0. Y locations and the lane segments are only used
        for the graphics output; they are not needed for the environment calculations, per se.

        CAUTION: This is not a general container.  This __init__ code defines the exact geometry of the
        scenario being used by this version of the code.
    """

    WIDTH = 20.0 #lane width, m; using a crazy large number so that grapics are pleasing
    COS_LANE2_ANGLE = 0.8660 #cosine of the angle of lane 2, segment 0, between the map frame and parametric frame

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

        # Lane 2 - two segments as the merge ramp; first seg is separate; second is adjacent to L1.
        # Segments show the lane at an angle to the main roadway, for visual appeal & clarity.
        L2_Y = L1_Y - Roadway.WIDTH
        segs = [(159.1, L2_Y-370.0,  800.0, L2_Y, 740.0),
                (800.0, L2_Y,       1320.0, L2_Y, 520.0)]
        lane = Lane(2, 159.1, 1260.0, segs, left_id = 1, left_join = 800.0, left_sep = 1320.0)
        self.lanes.append(lane)


    def map_to_param_frame(self,
                           x                : float,        #X coordinate in the map frame, m
                           lane             : int           #lane ID (0-indexed)
                          ) -> float:                       #Returns P coordinate, m
        """Converts a point in the map coordinate frame (x, y) to a corresponding point in the parametric coordinate
            frame (p, q). Since the vehicles have no freedom of lateral movement other than whole-lane changes, Y
            coordinates are not important, only lane IDs. These will not change between the frames.
        """

        p = x
        if lane == 2:
            join_point = self.lanes[2].segments[0][2]
            if x < join_point:
                p = join_point - (join_point - x)/self.COS_LANE2_ANGLE

        return p


    def param_to_map_frame(self,
                           p                : float,        #P coordinate in the parametric frame, m
                           lane             : int           #lane ID (0-indexed)
                          ) -> float:                       #Returns X coordinate, m
        """Converts a point in the parametric coordinate frame (p, q) to a corresponding point in the map frame (x, y).
            Since the vehicles have no freedom of lateral movement other than whole-lane changes, Q and Y coordinates
            are not important, only lane IDs, which will not change between coordinate frames.
        """

        x = p
        if lane == 2:
            join_point = self.lanes[2].segments[0][2]
            if p < join_point:
                x = max(join_point - (join_point - p)*self.COS_LANE2_ANGLE, self.lanes[2].start_x)

        return x


    def get_current_lane_geom(self,
                                lane_id         : int   = 0,    #ID of the lane in question
                                p_loc           : float = 0.0   #ego vehicle's P coordinate in the parametric frame, m
                             ) -> Tuple[float, int, float, float, float, int, float, float, float]:
        """Determines all of the variable roadway geometry relative to the given vehicle location.
            Returns a tuple of (remaining dist in this lane, m,
                                ID of left neighbor ln (or -1 if none),
                                dist to left adjoin point A, m,
                                dist to left adjoin point B, m,
                                remaining dist in left ajoining lane, m,
                                ID of right neighbor lane (or -1 if none),
                                dist to right adjoin point A, m,
                                dist to right adjoin point B, m,
                                remaining dist in right adjoining lane, m).
            If either adjoining lane doesn't exist, its return values will be 0, inf, inf, inf.  All distances are in m.
        """

        # Ensure that the given location is not prior to beginning of the lane
        assert self.param_to_map_frame(p_loc, lane_id) >= self.lanes[lane_id].start_x, \
                "///// Roadway.get_current_lane_geom: p_loc of {} is prior to beginning of lane {}".format(p_loc, lane_id)

        if self.debug > 1:
            print("///// Entering Roadway.get_current_lane_geom for lane_id = ", lane_id, ", p_loc = ", p_loc)
        rem_this_lane = self.lanes[lane_id].length - (p_loc - self.map_to_param_frame(self.lanes[lane_id].start_x, lane_id))

        la = 0.0
        lb = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        l_rem = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        left_id = self.lanes[lane_id].left_id
        if left_id >= 0:
            la = self.lanes[lane_id].left_join - p_loc
            lb = self.lanes[lane_id].left_sep - p_loc
            l_rem = self.lanes[left_id].length - (p_loc - self.map_to_param_frame(self.lanes[left_id].start_x, left_id))

        ra = 0.0
        rb = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        r_rem = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        right_id = self.lanes[lane_id].right_id
        if right_id >= 0:
            ra = self.lanes[lane_id].right_join - p_loc
            rb = self.lanes[lane_id].right_sep - p_loc
            r_rem = self.lanes[right_id].length - (p_loc - self.map_to_param_frame(self.lanes[right_id].start_x, right_id))

        if self.debug > 0:
            print("///// get_current_lane_geom complete. Returning rem = ", rem_this_lane)
            print("      lid = {}, la = {:.2f}, lb = {:.2f}, l_rem = {:.2f}".format(left_id, la, lb, l_rem))
            print("      rid = {}, ra = {:.2f}, rb = {:.2f}, r_rem = {:.2f}".format(right_id, ra, rb, r_rem))
        return rem_this_lane, left_id, la, lb, l_rem, right_id, ra, rb, r_rem


    def get_target_lane(self,
                        lane        : int,  #ID of the given lane
                        direction   : str,  #either "left" or "right"
                        p           : float #P coordinate for the location of interest, m
                       ) -> int:
        """Returns the lane ID of the adjacent lane on the indicated side of the given lane, or -1 if there is none
            currently adjoining.
        """

        if self.debug > 1:
            print("///// Entering Roadway.get_target_lane. lane = ", lane, ", direction = ", direction, ", p = ", p)
        assert 0 <= lane < len(self.lanes), "get_adjoining_lane_id input lane ID {} invalid.".format(lane)
        if direction != "left"  and  direction != "right":
            return -1

        # Find the adjacent lane ID, then if one exists ensure that current location is between the join & separation points.
        this_lane = self.lanes[lane]
        tgt = this_lane
        if direction == "left":
            tgt = this_lane.left_id
            if tgt >= 0:
                if p < this_lane.left_join  or  p > this_lane.left_sep:
                    tgt = -1

        else: #right
            tgt = this_lane.right_id
            if tgt >= 0:
                if p < this_lane.right_join  or  p > this_lane.right_sep:
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


    def get_lane_start_p(self,
                         lane   : int   #ID of the lane in question
                        ) -> float:
        """Returns the P coordinate of the beginning of the lane (in parametric frame)."""

        assert 0 <= lane < len(self.lanes), "Roadway.get_lane_start_p input lane ID {} invalid.".format(lane)
        return self.map_to_param_frame(self.lanes[lane].start_x, lane)



######################################################################################################
######################################################################################################



class Lane:
    """Defines the geometry of a single lane and its relationship to other lanes in the map frame.
        Note: an adjoining lane must join this one exactly once (possibly at distance 0), and
                may or may not separate from it farther downtrack. Once it separates, it cannot
                rejoin.  If it does not separate, then separation location will be same as this
                lane's length.
    """

    def __init__(self,
                    my_id       : int,                  #ID of this lane
                    start_x     : float,                #X coordinate of the start of the lane, m
                    length      : float,                #total length of this lane, m (includes any downtrack buffer)
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
                    tgt_speed   : float = 0.0,  #the (constant) target speed that the vehicle will try to maintain, m/s
                    cur_speed   : float = 0.0,  #vehicle's current speed, m/s
                    prev_speed  : float = 0.0,  #vehicle's speed in the previous time step, m/s
                    debug       : int = 0   #debug printing level
                ):

        self.time_step_size = step_size
        self.max_jerk = max_jerk
        self.tgt_speed = tgt_speed
        self.cur_speed = cur_speed
        self.prev_speed = prev_speed
        self.debug = debug

        self.lane_id = -1                   #-1 is an illegal value
        self.p = 0.0                        #P coordinate of vehicle center in parametric frame, m
        self.prev_accel = 0.0               #Forward actual acceleration in previous time step, m/s^2
        self.lane_change_status = "none"    #Initialized to no lane change underway
        self.active = True                  #is the vehicle an active part of the scenario? If false, it is invisible
        self.crashed = False                #has this vehicle crashed into another?


    def advance_vehicle_spd(self,
                            new_speed_cmd   : float,    #the newly commanded speed, m/s
                           ) -> Tuple[float, float]:
        """Advances a vehicle's forward motion for one time step according to the vehicle dynamics model.
            Note that this does not consider lateral motion, which needs to be handled elsewhere.

            Returns: tuple of (new speed (m/s), new P location (m))
        """

        # Determine the current & previous effective accelerations
        cur_accel_cmd = (new_speed_cmd - self.cur_speed) / self.time_step_size
        #print("///// Vehicle.advance_vehicle_spd: new_speed_cmd = {:.1f}, cur_speed = {:.1f}, prev_speed = {:.1f}, cur_accel_cmd = {:.2f}, prev_accel = {:.2f}"
        #      .format(new_speed_cmd, cur_speed, prev_speed, cur_accel_cmd, prev_accel))
        return self.advance_vehicle_accel(cur_accel_cmd)


    def advance_vehicle_accel(self,
                              new_accel_cmd   : float,    #newest fwd accel command, m/s^2
                             ) -> Tuple[float, float]:
        """Advances a vehicle's forward motion for one time step according to the vehicle dynamics model.
            Note that this does not consider lateral motion, which needs to be handled elsewhere.

            Returns: tuple of (new speed (m/s), new P location (m))
        """

        # Determine new jerk, accel, speed & location of the vehicle
        new_jerk = min(max((new_accel_cmd - self.prev_accel) / self.time_step_size, -self.max_jerk), self.max_jerk)
        new_accel = min(max(self.prev_accel + self.time_step_size*new_jerk, -SimpleHighwayRamp.MAX_ACCEL), SimpleHighwayRamp.MAX_ACCEL)
        new_speed = min(max(self.cur_speed + self.time_step_size*new_accel, 0.0), SimpleHighwayRamp.MAX_SPEED) #vehicle won't start moving backwards
        new_p = max(self.p + self.time_step_size*(new_speed + 0.5*self.time_step_size*new_accel), 0.0)

        # Update the state variables
        self.p = new_p
        self.prev_speed = self.cur_speed
        self.cur_speed = new_speed
        self.prev_accel = new_accel

        return new_speed, new_p


    def print(self,
                tag     : object = None     #tag to identify the vehicle
             ):
        """Prints the attributes of this vehicle object."""

        print("       [{}]: active = {:5}, lane_id = {:2d}, p = {:.2f}, status = {:5s}, speed = {:.2f}" \
                .format(tag, self.active, self.lane_id, self.p, self.lane_change_status, self.cur_speed))
