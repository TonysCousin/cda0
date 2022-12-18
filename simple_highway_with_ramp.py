from collections import deque
from statistics import mean
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

        Observation space:  In this version the agent magically knows everything going on in the environment, including
        some connectivities of the lane geometry.  Its observation space is described in the __init__() method (all float32).

        Action space:  continuous and contains the following elements in the vector (real world values, unscaled):
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

        Case 4, two parallel lanes indefinitely long:  no diagram needed, but A = 0 and B = inf, rem = inf.


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
        + If any vehicle drives off the end of a lane, it is a crash and ends the episode.
        + If a lane change is requested where no target lane exists, it is considered a crash and ends the episode.
        + If there is no crash, but all vehicles exit the indefinite end of a lane, then the episode is complete (success).

        Agent rewards are provided by a separate reward function.  The reward logic is documented there.
    """

    metadata = {"render_modes": None}
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    OBS_SIZE                = 29
    VEHICLE_LENGTH          = 5.0       #m
    MAX_SPEED               = 35.0      #vehicle's max achievable speed, m/s
    MAX_ACCEL               = 3.0       #vehicle's max achievable acceleration (fwd or backward), m/s^2
    MAX_JERK                = 4.0       #max desirable jerk for occupant comfort, m/s^3
    ROAD_SPEED_LIMIT        = 29.1      #Roadway's legal speed limit, m/s (29.1 m/s = 65 mph)
    SCENARIO_LENGTH         = 2000.0    #total length of the roadway, m
    SCENARIO_BUFFER_LENGTH  = 200.0     #length of buffer added to the end of continuing lanes, m
    #TODO: make this dependent upon time step size:
    HALF_LANE_CHANGE_STEPS  = 3.0 / 0.5 // 2    #num steps to get half way across the lane (equally straddling both)
    TOTAL_LANE_CHANGE_STEPS = 2 * HALF_LANE_CHANGE_STEPS
    MAX_STEPS_SINCE_LC      = 60        #largest num time steps we will track since previous lane change


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
            init_ego_speed: initial speed of the agent vehicle (defaults to random in [5, 20])
            init_ego_x:     initial downtrack location of the agent vehicle from lane begin (defaults to random in [0, 200])
        """

        super().__init__()

        # Store the arguments
        #self.seed = seed #Ray 2.0.0 chokes on the seed() method if this is defined (it checks for this attribute also)
        #TODO: try calling self.seed() without storing it as an instance attribute
        self.prng = np.random.default_rng(seed = seed)
        self.render_mode = render_mode

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

        self._set_initial_conditions(config)

        if self.debug > 0:
            print("\n///// SimpleHighwayRamp init: config = ", config)

        # Define the vehicles used in this scenario - the ego vehicle (where the AI agent lives) is index 0
        self.vehicles = []
        for i in range(4):
            v = Vehicle(self.time_step_size, SimpleHighwayRamp.MAX_JERK, self.debug)
            self.vehicles.append(v)

        #
        #..........Define the essential attributes required of any Env object: observation space and action space
        #

        # Indices into the observation vector (need to have all vehicles contiguous with ego being the first one)
        self.EGO_LANE_ID        =  0 #index of the lane the agent is occupying
        self.EGO_X              =  1 #agent's distance downtrack in that lane (center of bounding box), m
        self.EGO_SPEED          =  2 #agent's forward speed, m/s
        self.EGO_LANE_REM       =  3 #distance remaining in the agent's current lane, m
        self.N1_LANE_ID         =  4 #neighbor vehicle 1, index of the lane occupied by that vehicle
        self.N1_X               =  5 #neighbor vehicle 1, vehicle's dist downtrack in its current lane (center of bounding box), m
        self.N1_SPEED           =  6 #neighbor vehicle 1, vehicle's forward speed, m/s
        self.N1_LANE_REM        =  7 #neighbor vehicle 1, distance remaining in that vehicle's current lane, m
        self.N2_LANE_ID         =  8 #neighbor vehicle 2, index of the lane occupied by that vehicle
        self.N2_X               =  9 #neighbor vehicle 2, vehicle's dist downtrack in its current lane (center of bounding box), m
        self.N2_SPEED           = 10 #neighbor vehicle 2, vehicle's forward speed, m/s
        self.N2_LANE_REM        = 11 #neighbor vehicle 2, distance remaining in that vehicle's current lane, m
        self.N3_LANE_ID         = 12 #neighbor vehicle 3, index of the lane occupied by that vehicle
        self.N3_X               = 13 #neighbor vehicle 3, vehicle's dist downtrack in its current lane (center of bounding box), m
        self.N3_SPEED           = 14 #neighbor vehicle 3, vehicle's forward speed, m/s
        self.N3_LANE_REM        = 15 #neighbor vehicle 3, distance remaining in that vehicle's current lane, m
        self.EGO_ACCEL_CMD_CUR  = 16 #agent's most recent accel_cmd, m/s^2
        self.EGO_ACCEL_CMD_PREV1= 17 #agent's next most recent accel_cmd (1 time step old), m/s^2
        self.EGO_ACCEL_CMD_PREV2= 18 #agent's next most recent accel_cmd (2 time steps old), m/s^2
        self.EGO_LANE_CMD_CUR   = 19 #agent's most recent lane_change_cmd
        self.STEPS_SINCE_LN_CHG = 20 #num time steps since the previous lane change was initiated
        self.ADJ_LN_LEFT_ID     = 21 #index of the lane that is/will be adjacent to the left of ego lane (-1 if none)
        self.ADJ_LN_LEFT_CONN_A = 22 #dist from agent to where adjacent lane first joins ego lane, m
        self.ADJ_LN_LEFT_CONN_B = 23 #dist from agent to where adjacent lane separates from ego lane, m
        self.ADJ_LN_LEFT_REM    = 24 #dist from agent to end of adjacent lane, m
        self.ADJ_LN_RIGHT_ID    = 25 #index of the lane that is/will be adjacent to the right of ego lane (-1 if none)
        self.ADJ_LN_RIGHT_CONN_A= 26 #dist from agent to where adjacent lane first joins ego lane, m
        self.ADJ_LN_RIGHT_CONN_B= 27 #dist from agent to where adjacent lane separates from ego lane, m
        self.ADJ_LN_RIGHT_REM   = 28 #dist from agent to end of adjacent lane, m
        #Note:  lane IDs are always non-negative; if adj_ln_*_id is -1 then the other respective values on that side
        #       are meaningless, as there is no lane.
        #TODO future: replace this kludgy vehicle-specific observations with general obs on lane occupancy


        lower_obs = np.zeros((SimpleHighwayRamp.OBS_SIZE)) #most values are 0, so only the others are explicitly set below
        lower_obs[self.EGO_ACCEL_CMD_CUR]   = -SimpleHighwayRamp.MAX_ACCEL
        lower_obs[self.EGO_ACCEL_CMD_PREV1] = -SimpleHighwayRamp.MAX_ACCEL
        lower_obs[self.EGO_ACCEL_CMD_PREV2] = -SimpleHighwayRamp.MAX_ACCEL
        lower_obs[self.EGO_LANE_CMD_CUR]    = -1.0
        lower_obs[self.N1_LANE_ID]          = -1.0
        lower_obs[self.N2_LANE_ID]          = -1.0
        lower_obs[self.N3_LANE_ID]          = -1.0
        lower_obs[self.ADJ_LN_LEFT_ID]      = -1.0
        lower_obs[self.ADJ_LN_LEFT_CONN_A]  = -SimpleHighwayRamp.SCENARIO_LENGTH
        lower_obs[self.ADJ_LN_LEFT_CONN_B]  = -SimpleHighwayRamp.SCENARIO_LENGTH
        lower_obs[self.ADJ_LN_LEFT_REM]     = -SimpleHighwayRamp.SCENARIO_LENGTH
        lower_obs[self.ADJ_LN_RIGHT_ID]     = -1.0
        lower_obs[self.ADJ_LN_RIGHT_CONN_A] = -SimpleHighwayRamp.SCENARIO_LENGTH
        lower_obs[self.ADJ_LN_RIGHT_CONN_B] = -SimpleHighwayRamp.SCENARIO_LENGTH
        lower_obs[self.ADJ_LN_RIGHT_REM]    = -SimpleHighwayRamp.SCENARIO_LENGTH

        upper_obs = np.ones(SimpleHighwayRamp.OBS_SIZE)
        upper_obs[self.EGO_LANE_ID]         = 6.0
        upper_obs[self.EGO_X]               = SimpleHighwayRamp.SCENARIO_LENGTH
        upper_obs[self.EGO_SPEED]           = SimpleHighwayRamp.MAX_SPEED
        upper_obs[self.EGO_LANE_REM]        = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        upper_obs[self.N1_LANE_ID]          = 6.0
        upper_obs[self.N1_X]                = SimpleHighwayRamp.SCENARIO_LENGTH
        upper_obs[self.N1_SPEED]            = SimpleHighwayRamp.MAX_SPEED
        upper_obs[self.N1_LANE_REM]         = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        upper_obs[self.N2_LANE_ID]          = 6.0
        upper_obs[self.N2_X]                = SimpleHighwayRamp.SCENARIO_LENGTH
        upper_obs[self.N2_SPEED]            = SimpleHighwayRamp.MAX_SPEED
        upper_obs[self.N2_LANE_REM]         = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        upper_obs[self.N3_LANE_ID]          = 6.0
        upper_obs[self.N3_X]                = SimpleHighwayRamp.SCENARIO_LENGTH
        upper_obs[self.N3_SPEED]            = SimpleHighwayRamp.MAX_SPEED
        upper_obs[self.N3_LANE_REM]         = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        upper_obs[self.EGO_ACCEL_CMD_CUR]   = SimpleHighwayRamp.MAX_ACCEL
        upper_obs[self.EGO_ACCEL_CMD_PREV1] = SimpleHighwayRamp.MAX_ACCEL
        upper_obs[self.EGO_ACCEL_CMD_PREV2] = SimpleHighwayRamp.MAX_ACCEL
        upper_obs[self.EGO_LANE_CMD_CUR]    = 1.0
        upper_obs[self.STEPS_SINCE_LN_CHG]  = SimpleHighwayRamp.MAX_STEPS_SINCE_LC
        upper_obs[self.ADJ_LN_LEFT_ID]      = 6.0
        upper_obs[self.ADJ_LN_LEFT_CONN_A]  = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        upper_obs[self.ADJ_LN_LEFT_CONN_B]  = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        upper_obs[self.ADJ_LN_LEFT_REM]     = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        upper_obs[self.ADJ_LN_RIGHT_ID]     = 6.0
        upper_obs[self.ADJ_LN_RIGHT_CONN_A] = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        upper_obs[self.ADJ_LN_RIGHT_CONN_B] = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        upper_obs[self.ADJ_LN_RIGHT_REM]    = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH

        self.observation_space = Box(low=lower_obs, high=upper_obs, dtype=np.float32)
        if self.debug == 2:
            print("///// observation_space = ", self.observation_space)

        lower_act = np.array([-SimpleHighwayRamp.MAX_ACCEL, -1.0])
        upper_act = np.array([ SimpleHighwayRamp.MAX_ACCEL,  1.0])
        self.action_space = Box(low=lower_act, high=upper_act, dtype=np.float32)
        if self.debug == 2:
            print("///// action_space = ", self.action_space)

        self.obs = np.zeros(SimpleHighwayRamp.OBS_SIZE) #will be returned from reset() and step()
        self._verify_obs_limits("init after space defined")

        # Create the roadway geometry
        self.roadway = Roadway(self.debug)

        # Other persistent data
        self.lane_change_underway = "none" #possible values: "left", "right", "none"
        self.lane_change_count = 0 #num consecutive time steps since a lane change was begun
        self.steps_since_reset = 0 #length of the current episode in time steps
        self.stopped_count = 0 #num consecutive time steps in an episode where vehicle speed is zero
        self.reward_for_completion = True #should we award the episode completion bonus?
        self.episode_count = 0 #number of training episodes (number of calls to reset())
        self.accel_hist = deque(maxlen = 4)
        self.speed_hist = deque(maxlen = 4)


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


    def reset(self,
              seed:         int             = None,
              options:      dict            = None
             ) -> list:
        """Reinitializes the environment to prepare for a new episode.  This must be called before
            making any calls to step().

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!
        """

        if self.debug > 0:
            print("///// Entering reset")

        # We need the following line to seed self.np_random
        #super().reset(seed=seed) #apparently gym 0.26.1 doesn't implement this method in base class!
        #self.seed = seed #okay to pass it to the parent class, but don't store a local member copy!

        # options may be a dict that can specify additional configurations - unique to each particular env
        if options is not None:
            raise ValueError("reset() called with options, but options are not used in this environment.")

        # If we are in a training run, then choose widely randomized initial conditions
        ego_lane_id = None
        ego_x = None
        ego_speed = None
        if self.training:
            ego_lane_id = self._select_init_lane()
            ego_x = 0.0
            if self.randomize_start_dist:
                m = min(self.roadway.get_total_lane_length(ego_lane_id), SimpleHighwayRamp.SCENARIO_LENGTH) - 10.0
                max_distance = max(self.episode_count * (10.0 - m)/400000.0 + m, 10.0) #decreases over episodes
                ego_x = self.prng.random() * max_distance
            ego_speed = self.prng.random() * SimpleHighwayRamp.MAX_SPEED

        # Else, we are doing inference, so limit the randomness of the initial conditions
        else:
            ego_lane_id = int(self.prng.random()*3) if self.init_ego_lane is None  else  self.init_ego_lane
            ego_x = self.prng.random() * 200.0 if self.init_ego_x is None  else  self.init_ego_x
            ego_speed = self.prng.random() * 31.0 + 4.0 if self.init_ego_speed is None  else self.init_ego_speed

        # Neighbor vehicles always go a constant speed, always travel in lane 1, and always start at the same location
        self.vehicles[1].lane_id = 1
        self.vehicles[1].dist_downtrack = self.neighbor_start_loc + 4.0*SimpleHighwayRamp.VEHICLE_LENGTH #in front of vehicle n2
        self.vehicles[1].speed = self.neighbor_speed
        self.vehicles[1].lane_change_status = "none"
        self.vehicles[2].lane_id = 1
        self.vehicles[2].dist_downtrack = self.neighbor_start_loc + 2.0*SimpleHighwayRamp.VEHICLE_LENGTH #in front of vehicle n3
        self.vehicles[2].speed = self.neighbor_speed
        self.vehicles[2].lane_change_status = "none"
        self.vehicles[3].lane_id = 1
        self.vehicles[3].dist_downtrack = self.neighbor_start_loc + 0.0 #at end of the line of 3 neighbors
        self.vehicles[3].speed = self.neighbor_speed
        self.vehicles[3].lane_change_status = "none"
        if self.debug > 1:
            print("      in reset: vehicles = ")
            for i, v in enumerate(self.vehicles):
                v.print(i)

        # If the ego vehicle is in lane 1 (where the neighbor vehicles are), then we need to initialize its position so that it isn't
        # going to immediately crash with them (n1 is always the farthest downtrack and n3 is always the farthest uptrack). Give it
        # more room if going in front of the neighbors, as ego has limited accel and may be starting much slower than they are.
        if ego_lane_id == 1:
            min_loc = self.vehicles[3].dist_downtrack - 5.0*SimpleHighwayRamp.VEHICLE_LENGTH
            max_loc = self.vehicles[1].dist_downtrack + 20.0*SimpleHighwayRamp.VEHICLE_LENGTH
            if min_loc < ego_x < max_loc:
                ego_x = max_loc
                if self.debug > 0:
                    print("///// reset initializing agent to: lane = {}, speed = {:.2f}, x = {:.2f}".format(ego_lane_id, ego_speed, ego_x))
        #print("///// reset: training = {}, ego_lane_id = {}, ego_x = {:.2f}, ego_speed = {:.2f}".format(self.training, ego_lane_id, ego_x, ego_speed))
        self._verify_obs_limits("reset after initializing local vars")

        # Reinitialize the whole observation vector
        self.obs = np.zeros(SimpleHighwayRamp.OBS_SIZE)
        ego_rem, lid, la, lb, l_rem, rid, ra, rb, r_rem = self.roadway.get_current_lane_geom(ego_lane_id, ego_x)
        self.vehicles[0].lane_id = ego_lane_id
        self.vehicles[0].dist_downtrack = ego_x
        self.vehicles[0].speed = ego_speed
        self.vehicles[0].lane_change_status = "none"

        self.obs[self.EGO_LANE_ID]          = ego_lane_id
        self.obs[self.EGO_X]                = ego_x
        self.obs[self.EGO_SPEED]            = ego_speed
        self.obs[self.EGO_LANE_REM]         = ego_rem
        self.obs[self.ADJ_LN_LEFT_ID]       = lid
        self.obs[self.ADJ_LN_LEFT_CONN_A]   = la
        self.obs[self.ADJ_LN_LEFT_CONN_B]   = lb
        self.obs[self.ADJ_LN_LEFT_REM]      = l_rem
        self.obs[self.ADJ_LN_RIGHT_ID]      = rid
        self.obs[self.ADJ_LN_RIGHT_CONN_A]  = ra
        self.obs[self.ADJ_LN_RIGHT_CONN_B]  = rb
        self.obs[self.ADJ_LN_RIGHT_REM]     = r_rem
        self.obs[self.STEPS_SINCE_LN_CHG]   = SimpleHighwayRamp.MAX_STEPS_SINCE_LC
        self._verify_obs_limits("reset after populating main obs")

        self.obs[self.N1_LANE_ID]           = self.vehicles[1].lane_id
        self.obs[self.N1_X]                 = self.vehicles[1].dist_downtrack
        self.obs[self.N1_SPEED]             = self.vehicles[1].speed
        self.obs[self.N1_LANE_REM]          = self.roadway.get_total_lane_length(self.vehicles[1].lane_id) - self.vehicles[1].dist_downtrack
        self.obs[self.N2_LANE_ID]           = self.vehicles[2].lane_id
        self.obs[self.N2_X]                 = self.vehicles[2].dist_downtrack
        self.obs[self.N2_SPEED]             = self.vehicles[2].speed
        self.obs[self.N2_LANE_REM]          = self.roadway.get_total_lane_length(self.vehicles[2].lane_id) - self.vehicles[2].dist_downtrack
        self.obs[self.N3_LANE_ID]           = self.vehicles[3].lane_id
        self.obs[self.N3_X]                 = self.vehicles[3].dist_downtrack
        self.obs[self.N3_SPEED]             = self.vehicles[3].speed
        self.obs[self.N3_LANE_REM]          = self.roadway.get_total_lane_length(self.vehicles[3].lane_id) - self.vehicles[3].dist_downtrack

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
        return self.obs


    def step(self,
                action  : list      #list of floats; 0 = accel cmd, m/s^2, 1 = lane chg cmd (-1 left, 0 none, +1 right)
            ):
        """Executes a single time step of the environment.  Determines how the input actions will alter the
            simulated world and returns the resulting observations to the agent.

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!
        """

        if self.debug > 0:
            print("///// Entering step(): action = ", action)
            print("      vehicles array contains:")
            for i, v in enumerate(self.vehicles):
                v.print(i)
        assert -SimpleHighwayRamp.MAX_ACCEL <= action[0] <= SimpleHighwayRamp.MAX_ACCEL, "Input accel cmd invalid: {:.2f}".format(action[0])
        assert -1.0 <= action[1] <= 1.0, "Input lane change cmd is invalid: {:.2f}".format(action[1])

        self.steps_since_reset += 1

        #
        #..........Calculate new state for the ego vehicle and determine if episode is complete
        #

        done = False
        return_info = {"reason": "Unknown"}

        # Move all of the neighbor vehicles downtrack (ASSUMES all vehicles are represented contiguously in the obs vector).
        # This doesn't account for possible lane changes, which are handled seperately in the next section.
        for i, v in enumerate(self.vehicles):
            obs_idx = self.EGO_LANE_ID + 4*i
            lane_id = v.lane_id
            if self.debug > 1:
                print("      Advancing vehicle {} with obs_idx = {}, lane_id = {}".format(i, obs_idx, lane_id))
            new_accel_cmd = 0.0 #non-ego vehicles are constant speed for now
            prev_accel_cmd = 0.0
            if i == 0: #this is the ego vehicle
                new_accel_cmd = action[0]
                prev_accel_cmd = self.obs[self.EGO_ACCEL_CMD_CUR]
            new_speed, new_x = v.advance_vehicle(new_accel_cmd, prev_accel_cmd)
            if self.debug > 1:
                print("      Vehicle {} advanced with new_accel_cmd = {:.2f}. new_speed = {:.2f}, new_x = {:.2f}"
                        .format(i, new_accel_cmd, new_speed, new_x))
            new_rem, _, _, _, _, _, _, _, _ = self.roadway.get_current_lane_geom(lane_id, new_x)
            self.obs[obs_idx + 1] = new_x
            self.obs[obs_idx + 2] = new_speed
            self.obs[obs_idx + 3] = new_rem

        new_ego_speed = self.obs[self.EGO_SPEED]
        new_ego_x = self.obs[self.EGO_X]
        new_ego_rem = self.obs[self.EGO_LANE_REM]

        # If the ego vehicle has run off the end of the scenario, consider the episode successfully complete
        if new_ego_x >= SimpleHighwayRamp.SCENARIO_LENGTH:
            new_ego_x = SimpleHighwayRamp.SCENARIO_LENGTH #clip it so it doesn't violate obs bounds
            done = True
            return_info["reason"] = "Success; end of scenario"

        # Determine if we are beginning or continuing a lane change maneuver.
        # Accept a lane change command that lasts for several time steps or only one time step.  Once the first
        # command is received (when currently not in a lane change), then start the maneuver and ignore future
        # lane change commands until the underway maneuver is complete, which takes several time steps.
        # It's legal, but not desirable, to command opposite lane change directions in consecutive time steps.
        # TODO future: replace instance variables lane_change_underway and lane_id with those in vehicle[0]
        ran_off_road = False

        if action[1] < -0.5  or  action[1] > 0.5  or  self.lane_change_underway != "none":
            if self.lane_change_underway == "none": #count should always be 0 in this case, so initiate a new count
                if action[1] < -0.5:
                    self.lane_change_underway = "left"
                else:
                    self.lane_change_underway = "right"
                self.lane_change_count = 1
                if self.debug > 0:
                    print("      *** New lane change maneuver initiated. action[1] = {:.2f}, status = {}"
                            .format(action[1], self.lane_change_underway))
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

                # Else, we are still going; if we are exactly half-way then change the current lane ID & downtrack dist
                elif self.lane_change_count == SimpleHighwayRamp.HALF_LANE_CHANGE_STEPS:
                    new_ego_lane = tgt_lane
                    adjustment = self.roadway.adjust_downtrack_dist(int(self.obs[self.EGO_LANE_ID]), tgt_lane)
                    new_ego_x += adjustment
                    self.vehicles[0].dist_downtrack += adjustment

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
                    return_info["reason"] = "Ran off road; lane change got underway too late"
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
                return_info["reason"] = "Ran off road; in terminating lane"

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
        self.obs[self.EGO_ACCEL_CMD_CUR] = action[0]
        self.obs[self.EGO_LANE_CMD_CUR] = action[1]
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

        # Check that none of the vehicles have crashed into each other, accounting for a lane change in progress
        # taking up both lanes
        crash = False
        if not done:
            crash = self._check_for_collisions()
            done = crash
            if done:
                return_info["reason"] = "Crashed into neighbor vehicle"

        # If vehicle has been stopped for several time steps, then declare the episode done as a failure
        stopped_vehicle = False
        if self.vehicles[0].speed < 2.0:
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

        return self.obs, reward, done, return_info


    def close(self):
        """Closes any resources that may have been used during the simulation."""
        pass #this method not needed for this version


    ##### internal methods #####


    def _set_initial_conditions(self,
                                config:     EnvContext
                               ):
        """Sets the initial conditions of the ego vehicle in member variables (lane ID, speed, downtrack position)."""

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

        self.init_ego_x = None
        try:
            ex = config["init_ego_x"]
            if 0 <= ex < SimpleHighwayRamp.SCENARIO_LENGTH:
                self.init_ego_x = ex
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


    def _select_init_lane(self) -> int:
        """Chooses the initial lane for training runs, which may not be totally random."""

        if self.prng.random() < 0.5:
            return 2
        else:
            return int(self.prng.random()*2) #select 0 or 1


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
                    if abs(va.dist_downtrack - vb.dist_downtrack) <= SimpleHighwayRamp.VEHICLE_LENGTH:
                        if self.debug > 1:
                            print("      CRASH in same lane between vehicles {} and {} near {:.2f} m in lane {}"
                                    .format(i, j, va.dist_downtrack, va.lane_id))
                        crash = True
                        break

                # Else if they are in adjacent lanes, then
                elif abs(va.lane_id - vb.lane_id) == 1:

                    # If either vehicle is changing lanes at the moment, then
                    if va.lane_change_status != "none"  or  vb.lane_change_status != "none":

                        # If the lane changer's target lane belongs to the other vehicle, then
                        va_tgt = self.roadway.get_target_lane(va.lane_id, va.lane_change_status, va.dist_downtrack)
                        vb_tgt = self.roadway.get_target_lane(vb.lane_id, vb.lane_change_status, vb.dist_downtrack)
                        if va_tgt == vb.lane_id  or  vb_tgt == va.lane_id:

                            # Get adjusted downtrack distance of each vehcile relative to a common lane
                            vb_dist_in_lane_a = vb.dist_downtrack + self.roadway.adjust_downtrack_dist(vb.lane_id, va.lane_id)

                            # If the two are within a vehicle length of each other, then it's a crash
                            if abs(va.dist_downtrack - vb_dist_in_lane_a) <= SimpleHighwayRamp.VEHICLE_LENGTH:
                                if self.debug > 1:
                                    print("      CRASH in adjacent lanes between vehicles {} and {} near {:.2f} m in lane {}"
                                            .format(i, j, vb_dist_in_lane_a, va.lane_id))
                                crash = True
                                break

        #TODO: for early confidence only
        if crash:
            print("///// Collision detected between vehicles {} and {} at va lane = {}, va x = {:.1f}"
                    .format(i, j, va.lane_id, va.dist_downtrack))

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

            # If there was a single- or multi-car crash or then set a penalty, larger for multi-car crash
            if crash:
                reward = -50.0
                explanation = "Crashed into a vehicle. "

            elif off_road:
                reward = -40.0
                explanation = "Ran off road. "

            # Else if the vehicle just stopped in the middle of the road then
            elif stopped:

                # Subtract a penalty for no movement
                reward = -30.0
                explanation = "Vehicle stopped. "

            # Else (episode ended successfully)
            else:

                # If we are allowed to reward the completion bonus then add amount inversely proportional
                # to the length of the episode.
                if self.reward_for_completion:
                    #diff = 600 - self.steps_since_reset
                    #reward = max(47.26e-6 * diff*diff, 0.0)
                    reward = min(max(10.0 - 0.05882*(self.steps_since_reset - 130), 0.0), 10.0)
                    explanation = "Successful episode! {} steps".format(self.steps_since_reset)
                else:
                    reward = 0.0
                    explanation = "Completed episode, but no bonus due to rule violation."

        # Else, episode still underway
        else:

            """
            # If ego vehicle acceleration is jerky, then apply a penalty (worst case 0.003)
            jerk = (self.obs[self.EGO_ACCEL_CMD_CUR] - self.obs[self.EGO_ACCEL_CMD_PREV1]) / self.time_step_size
            penalty = 0.002 * jerk*jerk
            reward -= penalty
            if penalty > 0.0001:
                explanation += "Jerk pen {:.4f}. ".format(penalty)
            """

            # Penalty for exceeding roadway speed limit - in some cases, this triggers a cancellation of the
            # eventual completion award (similar punishment to stopping the episode, but without changing the
            # physical environment)
            norm_speed = self.obs[self.EGO_SPEED] / SimpleHighwayRamp.ROAD_SPEED_LIMIT #1.0 = speed limit
            penalty = 0.0
            if norm_speed > 1.0:
                diff = norm_speed - 1.0
                penalty = 10.0 * diff*diff
                explanation += "HIGH speed pen {:.4f}. ".format(penalty)
            elif norm_speed < 0.95:
                diff = 0.95 - norm_speed
                penalty = 0.4 * diff*diff
                explanation += "Low speed pen {:.4f}. ".format(penalty)
            reward -= penalty

            """
            # Track historical speed & accels for possible incentives
            self.accel_hist.append(self.obs[self.EGO_ACCEL_CMD_CUR])
            self.speed_hist.append(norm_speed)
            avg_accel = 0.0
            avg_speed = SimpleHighwayRamp.ROAD_SPEED_LIMIT
            if len(self.accel_hist) == self.accel_hist.maxlen:
                avg_accel = mean(self.accel_hist)
                avg_speed = mean(self.speed_hist)
            bonus = 0.0
            max_bonus = 0.15

            # If recent speed was above speed limit and avg accel since then was negative then
            if norm_speed > 1.02  and  self.obs[self.EGO_ACCEL_CMD_CUR] < 0.0:

                # Award a bonus proportional to the ratio of accel / excess speed
                slope = -self.obs[self.EGO_ACCEL_CMD_CUR]/SimpleHighwayRamp.MAX_ACCEL * max_bonus / (0.2 - 0.02)
                bonus = slope * (norm_speed - 1.02)

            # Else if recent speed was significantly below speed limit and avg accel since then was positive then
            elif norm_speed < 0.95  and  self.obs[self.EGO_ACCEL_CMD_CUR] > 0.0:

                # Award a bonus proportional to the ratio of accel / speed deficit
                intercept = self.obs[self.EGO_ACCEL_CMD_CUR] / SimpleHighwayRamp.MAX_ACCEL * max_bonus
                slope = -intercept / 0.95
                bonus = slope * norm_speed + intercept

            if bonus > 0.0001:
                reward += bonus
                explanation += "Accel bonus {:.4f}. ".format(bonus)
            #print("///// ep {}, step {}, speed {:.4f}, accel = {:.4f}, bonus = {:.4f}"
            #        .format(self.episode_count, self.steps_since_reset, norm_speed, self.obs[self.EGO_ACCEL_CMD_CUR], bonus))
            """

            # If a lane change was initiated, apply a penalty depending on how soon after the previous lane change
            if self.lane_change_count == 1:
                penalty = 0.1 + 0.01*(SimpleHighwayRamp.MAX_STEPS_SINCE_LC - self.obs[self.STEPS_SINCE_LN_CHG])
                reward -= penalty
                explanation += "Lane chg pen {:.4f}. ".format(penalty)

            # Penalty for lane change command not near one of the quantized action values
            lcc = self.obs[self.EGO_LANE_CMD_CUR]
            term = abs(lcc) - 0.5
            penalty = 0.0004*(1.0 - 4.0*term*term)
            reward -= penalty
            if penalty > 0.0001:
                explanation += "LCcmd pen {:.4f}. ".format(penalty)

        if self.debug > 0:
            print("///// reward returning {:.4f} due to crash = {}, off_road = {}, stopped = {}"
                    .format(reward, crash, off_road, stopped))

        return reward, explanation


    def _verify_obs_limits(self,
                           tag      : str = ""  #optional explanation of where in the code this was called
                          ):
        """Checks that each element of the observation vector is within the limits of the observation space."""

        lo = self.observation_space.low
        hi = self.observation_space.high

        try:
            for i in range(SimpleHighwayRamp.OBS_SIZE):
                assert lo[i] <= self.obs[i] <= hi[i], "\n///// obs[{}] value ({}) is outside bounds {} and {}" \
                                                        .format(i, self.obs[i], lo[i], hi[i])

        except AssertionError as e:
            print(e)
            print("///// Full obs vector content at {}:".format(tag))
            for j in range(SimpleHighwayRamp.OBS_SIZE):
                print("      {:2d}: {}".format(j, self.obs[j]))



######################################################################################################
######################################################################################################



class Roadway:
    """Defines the geometry of the roadway lanes and their drivable connections.

        CAUTION: This is not a general container.  This class code defines the exact geometry of the
        scenario being used by this version of the code.
    """

    def __init__(self,
                 debug      : int   #debug printing level
                ):

        self.debug = debug
        if self.debug > 1:
            print("///// Entering Roadway.__init__")
        self.lanes = [] #list of all the lanes in the scenario; list index is lane ID

        # The roadway being modeled looks roughly like the diagram at the top of this code file.
        really_long = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH

        # Lane 0 - single segment as the left through lane
        segs = [(0.0, 300.0, 2200.0, 300.0, 2200.0)]
        lane = Lane(0, really_long, segs,
                    right_id = 1, right_join = 0.0, right_sep = really_long)
        self.lanes.append(lane)

        # Lane 1 - single segment as the right through lane
        segs = [(0.0, 270.0, 2200.0, 270.0, 2200.0)]
        lane = Lane(1, really_long, segs,
                    left_id = 0, left_join = 0.0, left_sep = really_long,
                    right_id = 2, right_join = 800.0, right_sep = 1320.0)
        self.lanes.append(lane)

        # Lane 2 - two segments as the merge ramp; first seg is separate; second it adjacent to L1
        segs = [(384.3, 0.0,    800.0,  240.0, 480.0),
                (800.0, 240.0,  1320.0, 240.0, 520.0)]
        lane = Lane(2, 1000.0, segs, left_id = 1, left_join = 480.0, left_sep = 1000.0)
        self.lanes.append(lane)


    def get_current_lane_geom(self,
                                lane_id         : int   = 0,    #ID of the lane in question
                                dist_from_beg   : float = 0.0   #distance of ego vehicle from the beginning of the indicated lane, m
                             ) -> tuple:
        """Determines all of the variable roadway geometry relative to the given vehicle location.
            Returns a tuple of (remaining dist in this lane, ID of left neighbor ln (or -1 if none), dist to left adjoin point A,
                                dist to left adjoin point B, remaining dist in left ajoining lane, ID of right neighbor lane (or -1
                                if none), dist to right adjoin point A, dist to right adjoin point B,
                                remaining dist in right adjoining lane). If either adjoining lane doesn't exist, its return values
                                will be 0, inf, inf.  All distances are in m.
        """

        if self.debug > 1:
            print("///// Entering Roadway.get_current_lane_geom for lane_id = ", lane_id, ", dist = ", dist_from_beg)
        rem_this_lane = self.lanes[lane_id].length - dist_from_beg
        la = 0.0
        lb = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        l_rem = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        left_id = self.lanes[lane_id].left_id
        if left_id >= 0:
            la = self.lanes[lane_id].left_join - dist_from_beg
            lb = self.lanes[lane_id].left_sep - dist_from_beg
            l_rem = self.lanes[left_id].length - dist_from_beg - self.adjust_downtrack_dist(lane_id, left_id)
        ra = 0.0
        rb = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        r_rem = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        right_id = self.lanes[lane_id].right_id
        if right_id >= 0:
            ra = self.lanes[lane_id].right_join - dist_from_beg
            rb = self.lanes[lane_id].right_sep - dist_from_beg
            r_rem = self.lanes[right_id].length - dist_from_beg - self.adjust_downtrack_dist(lane_id, right_id)

        if self.debug > 0:
            print("///// get_current_lane_geom complete. Returning rem = ", rem_this_lane)
            print("      lid = {}, la = {:.2f}, lb = {:.2f}, l_rem = {:.2f}".format(left_id, la, lb, l_rem))
            print("      rid = {}, ra = {:.2f}, rb = {:.2f}, r_rem = {:.2f}".format(right_id, ra, rb, r_rem))
        return rem_this_lane, left_id, la, lb, l_rem, right_id, ra, rb, r_rem


    def adjust_downtrack_dist(self,
                                prev_lane_id    : int,
                                new_lane_id     : int
                             ):
        """Returns an adjustment to be applied to the downtrack distance in the current lane, as a result of changing lanes.
            A vehicle's downtrack distance is relative to the beginning of its current lane, and each lane may start at a
            different point.
            Return value can be positive or negative (float), m
        """

        if self.debug > 1:
            print("///// Entering Roadway.adjust_downtrack_dist. prev_lane = ", prev_lane_id, ", new_lane = ", new_lane_id)
        adjustment = 0.0

        # If new lane is on the right and prev lane is on the left, and they adjoin, then
        if self.lanes[new_lane_id].left_id == prev_lane_id  and  self.lanes[prev_lane_id].right_id == new_lane_id:
            adjustment = self.lanes[new_lane_id].left_join - self.lanes[prev_lane_id].right_join

        # Else if new lane is on the left and prev lane is on the right and the adjoin, then
        elif self.lanes[new_lane_id].right_id == prev_lane_id  and  self.lanes[prev_lane_id].left_id == new_lane_id:
            adjustment = self.lanes[new_lane_id].right_join - self.lanes[prev_lane_id].left_join

        if self.debug > 1:
            print("///// adjust_downtrack_dist complete. Returning ", adjustment)
        return adjustment


    def get_target_lane(self,
                        lane        : int,  #ID of the given lane
                        direction   : str,  #either "left" or "right"
                        distance    : float #distance downtrack in the given lane, m
                       ):
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
                             ):
        """Returns the total length of the requested lane, m"""

        assert 0 <= lane < len(self.lanes), "Roadway.get_total_lane_length input lane ID {} invalid.".format(lane)
        return self.lanes[lane].length



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
                    length      : float,                #total length of this lane, m (includes buffer)
                    segments    : list,                 #list of straight segments that make up this lane; each item is
                                                        # a tuple containing: (x0, y0, x1, y1, length), where
                                                        # x0, y0 are map coordinates of the starting point, in m
                                                        # x1, y1 are map coordinates of the ending point, in m
                                                        # length is the length of the segment, in m
                                                        #Each lane must have at least one segment, and segment lengths
                                                        # need to add up to total lane length
                    left_id     : int       = -1,       #ID of an adjoining lane to its left, or -1 for none
                    left_join   : float     = 0.0,      #dist downtrack where left lane first joins, m
                    left_sep    : float     = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH,
                                                        #dist downtrack where left lane separates from this one, m
                    right_id    : int       = -1,       #ID of an ajoining lane to its right, or -1 for none
                    right_join  : float     = 0.0,      #dist downtrack where right lane first joins, m
                    right_sep   : float     = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
                                                        #dist downtrack where right lane separates from this one, m
                ):

        self.my_id = my_id
        self.length = length
        self.left_id = left_id
        self.left_join = left_join
        self.left_sep = left_sep
        self.right_id = right_id
        self.right_join = right_join
        self.right_sep = right_sep
        self.segments = segments

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
            assert left_join >= 0.0  and  left_join < length, "Lane {} left_join value invalid.".format(my_id)
            assert left_sep > left_join, "Lane {} left_sep {} not larger than left_join {}".format(my_id, left_sep, left_join)
            assert left_sep <= length, "Lane {} left sep {} is larger than this lane's length.".format(my_id, left_sep)
        if right_id >= 0:
            assert right_id != my_id, "Lane {} right adjoining lane has same ID".format(my_id)
            assert right_join >= 0.0  and  right_join < length, "Lane {} right_join value invalid.".format(my_id)
            assert right_sep > right_join, "Lane {} right_sep {} not larger than right_join {}".format(my_id, right_sep, right_join)
            assert right_sep <= length, "Lane {} right sep {} is larger than this lane's length.".format(my_id, right_sep)
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

        self.lane_id = -1
        self.dist_downtrack = 0.0
        self.lane_change_status = "none"
        self.speed = 0.0


    def advance_vehicle(self,
                        new_accel_cmd   : float,    #newest fwd accel command, m/s^2
                        prev_accel_cmd  : float     #fwd accel command from prev time step, m/s^2
                       ):
        """Advances a vehicle's forward motion for one time step according to the vehicle dynamics model.
            Note that this does not consider lateral motion, which needs to be handled elsewhere.

            Returns: tuple of (new speed (m/s), new downtrack distance (m))
        """

        # Determine new jerk, accel, speed & downtrack distance of the ego vehicle
        new_jerk = (new_accel_cmd - prev_accel_cmd) / self.time_step_size
        if new_jerk < -self.max_jerk:
            new_jerk = -self.max_jerk
        elif new_jerk > self.max_jerk:
            new_jerk = self.max_jerk

        new_accel = min(max(prev_accel_cmd + self.time_step_size*new_jerk, -SimpleHighwayRamp.MAX_ACCEL), SimpleHighwayRamp.MAX_ACCEL)
        new_speed = min(max(self.speed + self.time_step_size*new_accel, 0.0), SimpleHighwayRamp.MAX_SPEED) #vehicle won't start moving backwards
        new_x = max(self.dist_downtrack + self.time_step_size*(new_speed + 0.5*self.time_step_size*new_accel), 0.0)

        self.dist_downtrack = new_x
        self.speed = new_speed

        return new_speed, new_x


    def print(self,
                tag     : object = None     #tag to identify the vehicle
             ):
        """Prints the attributes of this vehicle object."""

        print("       [{}]: lane_id = {:2d}, dist = {:.2f}, status = {:5s}, speed = {:.2f}" \
                .format(tag, self.lane_id, self.dist_downtrack, self.lane_change_status, self.speed))
