import numpy as np
from typing import Tuple
import gym
from ray.rllib.env.env_context import EnvContext

from simple_highway_with_ramp import SimpleHighwayRamp
from simple_highway_with_ramp import SimpleHighwayRamp

class SimpleHighwayRampWrapper(SimpleHighwayRamp):
    """Wraps the custom environment in order to properly convert observations and actions into usable structures for
        use by a torch NN.
    """

    def __init__(self,
                    config      : EnvContext
                ):

        super().__init__(config)

        self.debug = 0
        try:
            db = config["debug"]
        except KeyError as e:
            db = None
        if db is not None  and  db != ""  and  0 <= int(db) <= 2:
            self.debug = int(db)

        # Look for a flag indicating we are being used for training (vice inference)
        self.training = False
        try:
            tr = config["training"]
            if tr:
                self.training = True
        except KeyError as e:
            pass

        # Redefine the action space here, with properly scaled lower & upper limits (inherited from base class)
        self.action_space.low[0] /= SimpleHighwayRamp.MAX_ACCEL
        self.action_space.high[0] /= SimpleHighwayRamp.MAX_ACCEL


    def reset(self,
                seed    :   int = None,     #seed value for the PRNG
                options :   object = None   #currently not recognized by the Env, but appears for syntactic compliance
             ) -> np.array:                 #returns a scaled vector of observations usable by a NN

        """Invokes the environment's reset method, then scales the resulting observations to be usable by a NN."""

        obs = super().reset(seed, options)
        return self.scale_obs(obs)


    def step(self,
                action  :   list            #list of actions output from an NN
            ) -> Tuple[np.array, list, list, dict]: #returns scaled obs, rewards, dones and infos, where obs are scaled for NN consumption

        """Unscales the input actions from NN compliance to the 'raw' scale expected by the environment, then
            passes these actions to the environment to advance it one step and to gather new observations and rewards.

            If the "training" config param is True, then the return obs  scales the resulting observations,
            such that it will be usable as input to a NN.  The rewards, dones and info structures are not modfied.
            However, if the "training" config does not exist or is not True, then the returned obs list is NOT scaled.
            This allows an inference engine to directly interpret the observations.  It will then be responsible for
            passing that unscaled obs structure into the scale_obs() method to transform it into values that can be
            sent back to the NN for the next time step.
        """

        a = self._unscale_actions(action)
        raw_obs, r, d, i = super().step(a)
        if self.training:
            o = self.scale_obs(raw_obs)
        else:
            o = raw_obs


        #TODO debug only
        #print("///// wrapper.step: scaled obs vector =")
        #for j in range(len(o)):
        #    print("      {:2d}:  {}".format(j, o[j]))


        return o, r, d, i


    def scale_obs(self,
                    obs     : np.array  #raw observation vector from the environment
                  ) -> np.array:        #returns obs vector scaled for use by NN

        """Converts a raw observation vector from the parent environment to a scaled vector usable by a NN."""

        scaled = [None]*SimpleHighwayRamp.OBS_SIZE

        # An NN can't do anything with lane IDs directly.  The best we can tell it (in this version) is that something
        # is happening in the agent's own lane, or in the lane immediately to its left or in the lane immediately to its
        # right.  Therefore, these translations are specific to the roadway geometry hard-coded in this version.

        # Determine relative lane indexes for the neighbor vehicles (-1 is left of agent, 0 is same as agent, +1 is right of
        # agent; neighbors are always in lane 1)
        ego_lane_id = obs[self.EGO_LANE_ID]
        if ego_lane_id == 0:
            neighbor_lane = 1
        elif ego_lane_id == 1:
            neighbor_lane = 0
        else:
            neighbor_lane = -1

        scaled[self.EGO_LANE_ID]        = ego_lane_id
        scaled[self.EGO_X]              = obs[self.EGO_X]               / SimpleHighwayRamp.SCENARIO_LENGTH     #range [0, 1]
        scaled[self.EGO_SPEED]          = obs[self.EGO_SPEED]           / SimpleHighwayRamp.MAX_SPEED           #range [0, 1]
        scaled[self.EGO_LANE_REM]       = min(obs[self.EGO_LANE_REM]    / SimpleHighwayRamp.SCENARIO_LENGTH, 1.1) #range [0, 1.1]
        scaled[self.N1_LANE_ID]         = neighbor_lane
        scaled[self.N1_X]               = obs[self.N1_X]                / SimpleHighwayRamp.SCENARIO_LENGTH     #range [0, 1]
        scaled[self.N1_SPEED]           = obs[self.N1_SPEED]            / SimpleHighwayRamp.MAX_SPEED           #range [0, 1]
        scaled[self.N1_LANE_REM]        = min(obs[self.N1_LANE_REM]     / SimpleHighwayRamp.SCENARIO_LENGTH, 1.1) #range [0, 1.1]
        scaled[self.N2_LANE_ID]         = neighbor_lane
        scaled[self.N2_X]               = obs[self.N2_X]                / SimpleHighwayRamp.SCENARIO_LENGTH     #range [0, 1]
        scaled[self.N2_SPEED]           = obs[self.N2_SPEED]            / SimpleHighwayRamp.MAX_SPEED           #range [0, 1]
        scaled[self.N2_LANE_REM]        = min(obs[self.N2_LANE_REM]     / SimpleHighwayRamp.SCENARIO_LENGTH, 1.1) #range [0, 1.1]
        scaled[self.N3_LANE_ID]         = neighbor_lane
        scaled[self.N3_X]               = obs[self.N3_X]                / SimpleHighwayRamp.SCENARIO_LENGTH     #range [0, 1]
        scaled[self.N3_SPEED]           = obs[self.N3_SPEED]            / SimpleHighwayRamp.MAX_SPEED           #range [0, 1]
        scaled[self.N3_LANE_REM]        = min(obs[self.N3_LANE_REM]     / SimpleHighwayRamp.SCENARIO_LENGTH, 1.1) #range [0, 1.1]
        scaled[self.EGO_ACCEL_CMD_CUR]  = obs[self.EGO_ACCEL_CMD_CUR]   / SimpleHighwayRamp.MAX_ACCEL           #range [-1, 1]
        scaled[self.EGO_ACCEL_CMD_PREV1]= obs[self.EGO_ACCEL_CMD_PREV1] / SimpleHighwayRamp.MAX_ACCEL           #range [-1, 1]
        scaled[self.EGO_ACCEL_CMD_PREV2]= obs[self.EGO_ACCEL_CMD_PREV2] / SimpleHighwayRamp.MAX_ACCEL           #range [-1, 1]
        scaled[self.EGO_LANE_CMD_CUR]   = obs[self.EGO_LANE_CMD_CUR] #range [-1, 1]
        scaled[self.STEPS_SINCE_LN_CHG] = obs[self.STEPS_SINCE_LN_CHG]  / SimpleHighwayRamp.MAX_STEPS_SINCE_LC  #range [0, 1]
        # ADJ_LN_LEFT_ID is replaced with a boolean (0=false, 1=true) to indicate whether a left neighbor lane exists
        scaled[self.ADJ_LN_LEFT_ID] = 1 if obs[self.ADJ_LN_LEFT_ID] >= 0 else 0
        scaled[self.ADJ_LN_LEFT_CONN_A] = min(max(obs[self.ADJ_LN_LEFT_CONN_A] / SimpleHighwayRamp.SCENARIO_LENGTH,
                                                    -1.0), 1.1) #range [-1, 1.1]
        scaled[self.ADJ_LN_LEFT_CONN_B] = min(obs[self.ADJ_LN_LEFT_CONN_B] / SimpleHighwayRamp.SCENARIO_LENGTH, 1.1) #range [0, 1.1]
        scaled[self.ADJ_LN_LEFT_REM]    = min(obs[self.ADJ_LN_LEFT_REM] / SimpleHighwayRamp.SCENARIO_LENGTH, 1.1) #range [0, 1.1]
        # ADJ_LN_RIGHT_ID is replaced with a boolean (0=false, 1=true) to indicate whether a right neighbor lane exists
        scaled[self.ADJ_LN_RIGHT_ID] = 1 if obs[self.ADJ_LN_RIGHT_ID] >= 0 else 0
        scaled[self.ADJ_LN_RIGHT_CONN_A]= min(max(obs[self.ADJ_LN_RIGHT_CONN_A] / SimpleHighwayRamp.SCENARIO_LENGTH,
                                                    -1.0), 1.1) #range [-1, 1.1]
        scaled[self.ADJ_LN_RIGHT_CONN_B]= min(obs[self.ADJ_LN_RIGHT_CONN_B] / SimpleHighwayRamp.SCENARIO_LENGTH, 1.1) #range [0, 1.1]
        scaled[self.ADJ_LN_RIGHT_REM]   = min(obs[self.ADJ_LN_RIGHT_REM] / SimpleHighwayRamp.SCENARIO_LENGTH, 1.1) #range [0, 1.1]

        # Return the obs as an ndarray
        vec = np.array(scaled, dtype=np.float32)
        if self.debug > 1:
            print("_scale_obs returning vec size = ", vec.shape)
            print(vec)

        return vec


    ##### internal methods #####

    def _unscale_actions(self,
                            actions : list  #scaled actions generated by a NN
                        ) -> list:          #returns list of unscaled actions usable by the environment

        """Converts a list of action values from scaled, NN compliance to the scale expected by the parent environment."""

        al = SimpleHighwayRamp.MAX_ACCEL
        unscaled = [None]*2
        unscaled[0] = min(max(actions[0] * al, -al), al)    #accel command, m/s^2
        unscaled[1] = min(max(actions[1], -1.0), 1.0)       #lane change command

        return unscaled
