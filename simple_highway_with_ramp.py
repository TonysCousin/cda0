from cmath import inf
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

    OBS_SIZE        = 30
    VEHICLE_LENGTH  = 5.0   #m
    MAX_ACCEL       = 3.0   #m/s^2


    def __init__(self, config: EnvContext, seed=None, render_mode=None):
        """Initialize an object of this class."""

        #print("\n///// SimpleHighwayRamp init: config = ", config)

        # Store the arguments
        #self.seed = seed #Ray 2.0.0 chokes on the seed() method if this is defined (it checks for this attribute also)
        #TODO: try calling self.seed() without storing it as an instance attribute
        self.prng = np.random.default_rng(seed=seed)

        self.render_mode = render_mode
        self.time_step_size = 0.5 #seconds
        if config["time_step_size"] is not None  and  config["time_step_size"] != ""  and  float(config["time_step_size"]) > 0.0:
            self.time_step_size = float(config["time_step_size"])
        print("///// Environment is using time step size of {:.2f} sec")

        # Define the essential attributes required of any Env object: observation space and action space

        # Indices into the observation vector
        self.EGO_LANE_ID        =  0 #index of the lane the agent is occupying
        self.EGO_X              =  1 #agent's distance downtrack in that lane (center of bounding box), m
        self.EGO_SPEED          =  2 #agent's forward speed, m/s
        self.EGO_LANE_REM       =  3 #distance remaining in the agent's current lane, m
        self.EGO_ACCEL_CMD_CUR  =  4 #agent's most recent accel_cmd, m/s^2
        self.EGO_ACCEL_CMD_PREV1=  5 #agent's next most recent accel_cmd (1 time step old), m/s^2
        self.EGO_ACCEL_CMD_PREV2=  6 #agent's next most recent accel_cmd (2 time steps old), m/s^2
        self.EGO_LANE_CMD_CUR   =  7 #agent's most recent lane_change_cmd
        self.EGO_LANE_CMD_PREV1 =  8 #agent's next most recent lane_change_cmd (1 time step old)
        self.EGO_LANE_CMD_PREV2 =  9 #agent's next most recent lane_change_cmd (2 time steps old)
        self.ADJ_LN_LEFT_ID     = 10 #index of the lane that is/will be adjacent to the left of ego lane (-1 if none)
        self.ADJ_LN_LEFT_CONN_A = 11 #dist from agent to where adjacent lane first joins ego lane, m
        self.ADJ_LN_LEFT_CONN_B = 12 #dist from agent to where adjacent lane separates from ego lane, m
        self.ADJ_LN_LEFT_REM    = 13 #dist from agent to end of adjacent lane, m
        self.ADJ_LN_RIGHT_ID    = 14 #index of the lane that is/will be adjacent to the right of ego lane (-1 if none)
        self.ADJ_LN_RIGHT_CONN_A= 15 #dist from agent to where adjacent lane first joins ego lane, m
        self.ADJ_LN_RIGHT_CONN_B= 16 #dist from agent to where adjacent lane separates from ego lane, m
        self.ADJ_LN_RIGHT_REM   = 17 #dist from agent to end of adjacent lane, m
        self.N1_LANE_ID         = 18 #neighbor vehicle 1, index of the lane occupied by that vehicle
        self.N1_X               = 19 #neighbor vehicle 1, vehicle's dist downtrack in its current lane (center of bounding box), m
        self.N1_SPEED           = 20 #neighbor vehicle 1, vehicle's forward speed, m/s
        self.N1_LANE_REM        = 21 #neighbor vehicle 1, distance remaining in that vehicle's current lane, m
        self.N2_LANE_ID         = 22 #neighbor vehicle 2, index of the lane occupied by that vehicle
        self.N2_X               = 23 #neighbor vehicle 2, vehicle's dist downtrack in its current lane (center of bounding box), m
        self.N2_SPEED           = 24 #neighbor vehicle 2, vehicle's forward speed, m/s
        self.N2_LANE_REM        = 25 #neighbor vehicle 2, distance remaining in that vehicle's current lane, m
        self.N3_LANE_ID         = 26 #neighbor vehicle 3, index of the lane occupied by that vehicle
        self.N3_X               = 27 #neighbor vehicle 3, vehicle's dist downtrack in its current lane (center of bounding box), m
        self.N3_SPEED           = 28 #neighbor vehicle 3, vehicle's forward speed, m/s
        self.N3_LANE_REM        = 29 #neighbor vehicle 3, distance remaining in that vehicle's current lane, m
        #Note:  lane IDs are always non-negative; if adj_ln_*_id is -1 then the other respective values on that side
        #       are meaningless, as there is no lane.
        #TODO: replace this kludgy vehicle-specific observations with general obs on lane occupancy


        lower_obs = np.zeros((30)) #most values are 0, so only the others are explicitly described here
        lower_obs[4] = lower_obs[5] = lower_obs[6] = -SimpleHighwayRamp.MAX_ACCEL #historical ego acceleration cmds
        lower_obs[7] = lower_obs[8] = lower_obs[9] = -1.0 #historical ego lane cmds

        upper_obs = np.array([  6.0,    #ego_lane_id
                                2000.0, #ego_x
                                35.0,   #ego_speed
                                3000.0, #ego_lane_rem
                                SimpleHighwayRamp.MAX_ACCEL,    #ego_accel_cmd_cur
                                SimpleHighwayRamp.MAX_ACCEL,    #ego_accel_cmd_prev1
                                SimpleHighwayRamp.MAX_ACCEL,    #ego_accel_cmd_prev2
                                1.0,    #ego_lane_cmd_cur
                                1.0,    #ego_lane_cmd_prev1
                                1.0,    #ego_lane_cmd_prev2
                                6.0,    #adj_ln_left_id
                                3000.0, #adj_ln_left_conn_a
                                3000.0, #adj_ln_left_conn_b
                                3000.0, #adj_ln_left_rem
                                6.0,    #adj_ln_right_id
                                3000.0, #adj_ln_right_conn_a
                                3000.0, #adj_ln_right_conn_b
                                3000.0, #adj_ln_right_rem
                                6.0,    #n1_lane_id
                                2000.0, #n1_x
                                35.0,   #n1_speed
                                3000.0, #n1_lane_rem
                                6.0,    #n2_lane_id
                                2000.0, #n2_x
                                35.0,   #n2_speed
                                3000.0, #n2_lane_rem
                                6.0,    #n3_lane_id
                                2000.0, #n3_x
                                35.0,   #n3_speed
                                3000.0 #n3_lane_rem
                            ])

        self.observation_space = Box(low=lower_obs, high=upper_obs, shape=(self.OBS_SIZE), dtype=np.float32)

        lower_act = np.array([-SimpleHighwayRamp.MAX_ACCEL, -1.0])
        upper_act = np.array([ SimpleHighwayRamp.MAX_ACCEL,  1.0])
        self.action_space = Box(low=lower_act, high=upper_act, shape=(2), dtype=np.float32)

        self.obs = np.zeros(SimpleHighwayRamp.OBS_SIZE) #will be returned from reset() and step()

        # Create the roadway geometry
        self.roadway = Roadway

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

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        #self.seed = seed #okay to pass it to the parent class, but don't store a local member copy!

        # options may be a dict that can specify additional configurations - unique to each particular env
        if options is not None:
            raise ValueError("reset() called with options, but options are not used in this environment.")

        # Initialize a new set of observations
        self.obs = np.zeros(SimpleHighwayRamp.OBS_SIZE)
        ego_x = self.prng.random() * 200.0   #starting downtrack distance, m
        ego_rem, la, lb, l_rem, ra, rb, r_rem = self._get_current_lane_geom(0, ego_x)
        #future (all 3): n1_rem, _, _, _, _, _, _ = self._get_current_lane_geom(n1_lane, n1_x)
        self.obs[self.EGO_X]                = ego_x
        self.obs[self.EGO_SPEED]            = self.prng.random() * 15.0 + 5.0 #starting speed between 5 and 20 m/s
        self.obs[self.EGO_LANE_REM]         = ego_rem
        self.obs[self.ADJ_LN_LEFT_ID]       = 1 #fixed starting condition for this roadway scenario
        self.obs[self.ADJ_LN_LEFT_CONN_A]   = la
        self.obs[self.ADJ_LN_LEFT_CONN_B]   = lb
        self.obs[self.ADJ_LN_LEFT_REM]      = l_rem
        self.obs[self.ADJ_LN_RIGHT_ID]      = -1
        self.obs[self.ADJ_LN_RIGHT_CONN_A]  = ra
        self.obs[self.ADJ_LN_RIGHT_CONN_B]  = rb
        self.obs[self.ADJ_LN_RIGHT_REM]     = r_rem
        #turning off neighbor vehicles for initial phase - since these are constant speed, they stay out of the way
        self.obs[self.N1_LANE_ID]           = 1
        self.obs[self.N1_X]                 = 4.0 * SimpleHighwayRamp.VEHICLE_LENGTH #in front of vehicle n2
        self.obs[self.N1_SPEED]             = 0.0
        self.obs[self.N2_LANE_ID]           = 1
        self.obs[self.N2_X]                 = 2.0 * SimpleHighwayRamp.VEHICLE_LENGTH #in front of vehicle n3
        self.obs[self.N2_SPEED]             = 0.0
        self.obs[self.N3_LANE_ID]           = 1
        self.obs[self.N3_X]                 = 0.0 #at beginning of lane
        self.obs[self.N3_SPEED]             = 0.0

        return self.obs


    def step(self, action):
        """Executes a single time step of the environment.  Determines how the input actions will alter the
            simulated world and returns the resulting observations to the agent.

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!
        """
        assert action[0] in [0, 1], action
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
        return self.obs, reward, done, {}


    def close(self):
        """Closes any resources that may have been used during the simulation."""
        pass


    ##### internal methods #####

    def _get_current_lane_geom(lane_id, dist_from_beg):
        """Determines all of the variable roadway geometry relative to the given vehicle location."""

        return 0, 0, 0, 0, 0, 0, 0 #TODO: bogus!!!

######################################################################################################
######################################################################################################

class Roadway:
    """Defines the geometry of the roadway lanes and their drivable connections.
        This is not a general container.  This class code defines the exact geometry of the
        scenario being used by this version of the code.
    """

    SCENARIO_LENGTH = 2000.0    #length of the longest possible drive in an episode, m
    ONGOING_PAD     = 1000.0    #length of pad added to the end of "indefinite" lanes, m
                                # allows cars approaching end of episode to behave as if they will
                                # continue driving indefinitely farther

    def __init__(self):

        self.lanes = [] #list of all the lanes in the scenario; list index is lane ID

        # The roadway being modeled looks roughly like the diagram at the top of this code file.
        lane = Lane(0, Roadway.SCENARIO_LENGTH + Roadway.ONGOING_PAD,
                    right_id = 1, right_join = 0.0, right_sep = inf)
        self.lanes.append(lane)








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
                    left_id     : int       = -1,       #ID of an adjoining lane to its left, or -1 for none
                    left_join   : float     = 0.0,      #dist downtrack where left lane first joins, m
                    left_sep    : float     = inf,      #dist downtrack where left lane separates from this one, m
                    right_id    : int       = -1,       #ID of an ajoining lane to its right, or -1 for none
                    right_join  : float     = 0.0,      #dist downtrack where right lane first joins, m
                    right_sep   : float     = inf       #dist downtrack where right lane separates from this one, m
                ):

        self.my_id = my_id
        self.length = length
        self.left_id = left_id
        self.left_join = left_join
        self.left_sep = left_sep
        self.right_id = right_id
        self.right_join = right_join
        self.right_sep = right_sep

        assert length > 0.0, "Lane {} length of {} is invalid.".format(my_id, length)
        if left_id >= 0:
            assert left_id != my_id, "Lane {} left adjoining lane has same ID".format(my_id)
            assert left_join >= 0.0  and  left_join < length, "Lane {} left_join value invalid.".format(my_id)
            assert left_sep > left_join, "Lane {} left_sep {} not larger than left_join {}".format(my_id, left_sep, left_join)
            if left_sep < inf:
                assert left_sep <= length, "Lane {} left sep {} is larger than this lane's length.".format(my_id, left_sep)
        if right_id >= 0:
            assert right_id != my_id, "Lane {} right adjoining lane has same ID".format(my_id)
            assert right_join >= 0.0  and  right_join < length, "Lane {} right_join value invalid.".format(my_id)
            assert right_sep > right_join, "Lane {} right_sep {} not larger than right_join {}".format(my_id, right_sep, right_join)
            if right_sep < inf:
                assert right_sep <= length, "Lane {} right sep {} is larger than this lane's length.".format(my_id, right_sep)
        if left_id >= 0  and  right_id >= 0:
            assert left_id != right_id, "Lane {}: both left and right adjoining lanes share ID {}".format(my_id, left_id)


    def get_dist_to_join(self,
                            current_loc : float,        #dist downtrack in this lane of the point of interest, m
                            left_side   : bool          #is the request for a lane on the left? False = right side
                        ):
        """Returns the distance from current_loc to nearest joining lane on the specified side."""

        pass #TODO


    def get_dist_to_separation(self,
                                current_loc : float,    #dist downtrack in this lane of the point of interest, m
                                left_side   : bool      #is the request for a lane on the left? False = right side
                              ):
        """Returns the distance from current_loc to nearest separation of an adjoining lane on the specified side.
            Note that the separating lane may no yet be adjoining at current_locc.
        """

        pass #TODO
