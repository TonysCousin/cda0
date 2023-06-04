import numpy as np
from typing import Tuple, Dict
from ray.rllib.env.env_context import EnvContext
from gymnasium.spaces import Box

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


    def unscaled_reset(self, *,
                seed    :   int = None,     #seed value for the PRNG
                options :   object = None   #currently not recognized by the Env, but appears for syntactic compliance
             ) -> Tuple[np.array, dict]:    #returns a scaled vector of observations usable by a NN plus an info dict

        """Invokes the environment's reset method, but does not scale the resulting observations. This supports an
            inference engine using the output directly in real world units.
            CAUTION: intended to be used only for initial reset by an inference engine. The output will need to be
                     scaled externally if it is to be fed into an NN.
        """

        obs, info = super().reset(options = options)
        return obs, info


    def reset(self, *,
                seed    :   int = None,     #seed value for the PRNG
                options :   object = None   #currently not recognized by the Env, but appears for syntactic compliance
             ) -> Tuple[np.array, dict]:    #returns a scaled vector of observations usable by a NN plus an info dict

        """Invokes the environment's reset method, then scales the resulting observations to be usable by a NN."""

        obs, info = super().reset(options = options)
        return self.scale_obs(obs), info


    def step(self,
                action  :   list                            #list of actions output from an NN
            ) -> Tuple[np.array, float, bool, bool, Dict]:  #returns scaled obs, rewards, dones truncated flag, and infos,
                                                            # where obs are scaled for NN consumption

        """Passes the actions (after unscaling) to the environment to advance it one step and to gather new observations and rewards.

            If the "training" config param is True, then the return obs needs the resulting observations scaled,
            such that it will be usable as input to a NN.  The rewards, dones and info structures are not modfied.
            However, if the "training" config does not exist or is not True, then the returned obs list is NOT scaled.
            This allows an inference engine to directly interpret the observations.  It will then be responsible for
            passing that unscaled obs structure into the scale_obs() method to transform it into values that can be
            sent back to the NN for the next time step, if that action is to be taken.
        """

        # Unscale the action values
        #ua = [None]*2
        #ua[0] = action[0] * SimpleHighwayRamp.MAX_SPEED #Desired speed, m/s
        #ua[1] = math.floor(action[1] + 0.5) + 1.0       #maps desired lane from [-1, 1] into (0, 1, 2)
        #print("///// WRAPPER step: action = ", action, ", ua = ", ua)

        # Step the environment
        raw_obs, r, d, t, i = super().step(action)
        if self.training:
            o = self.scale_obs(raw_obs)
        else:
            o = raw_obs

        #print("///// wrapper.step: scaled obs vector =")
        #for j in range(len(o)):
        #    print("      {:2d}:  {}".format(j, o[j]))

        return o, r, d, t, i


    def scale_obs(self,
                    obs     : np.array  #raw observation vector from the environment
                  ) -> np.array:        #returns obs vector scaled for use by NN

        """Converts a raw observation vector from the parent environment to a scaled vector usable by a NN."""

        scaled = [None]*self.OBS_SIZE

        scaled[self.EGO_LANE_ID]        = obs[self.EGO_LANE_ID] - 1.0                                           #maps {0, 1, 2} to {-1, 0, 1}
        scaled[self.EGO_P]              = obs[self.EGO_P]               / SimpleHighwayRamp.SCENARIO_LENGTH     #range [0, 1]
        scaled[self.EGO_LANE_REM]       = min(obs[self.EGO_LANE_REM]    / SimpleHighwayRamp.SCENARIO_LENGTH, 1.1) #range [0, 1.1]
        scaled[self.EGO_SPEED]          = obs[self.EGO_SPEED]           / SimpleHighwayRamp.MAX_SPEED           #range [0, 1]
        scaled[self.EGO_SPEED_PREV]     = obs[self.EGO_SPEED_PREV]      / SimpleHighwayRamp.MAX_SPEED           #range [0, 1]
        scaled[self.STEPS_SINCE_LN_CHG] = obs[self.STEPS_SINCE_LN_CHG]  / SimpleHighwayRamp.MAX_STEPS_SINCE_LC  #range [0, 1]
        scaled[self.NEIGHBOR_IN_EGO_ZONE] = obs[self.NEIGHBOR_IN_EGO_ZONE]
        scaled[self.EGO_DES_SPEED]      = obs[self.EGO_DES_SPEED]       / SimpleHighwayRamp.MAX_SPEED           #range [0, 1]
        scaled[self.EGO_DES_SPEED_PREV] = obs[self.EGO_DES_SPEED_PREV]  / SimpleHighwayRamp.MAX_SPEED           #range [0, 1]
        scaled[self.EGO_DES_LN]         = obs[self.EGO_DES_LN] - 1.0                                            #maps {0, 1, 2} to {-1, 0, 1}
        scaled[self.EGO_DES_LN_PREV]    = obs[self.EGO_DES_LN_PREV] - 1.0                                       #maps {0, 1, 2} to {-1, 0, 1}

        # Handle all the geometric zones - none of these need scaling at this time, so just copy each element
        base = self.Z1_DRIVEABLE
        zone_data_size = self.Z2_DRIVEABLE - self.Z1_DRIVEABLE
        for zone in range(9):
            offset = base + zone*zone_data_size
            for i in range(zone_data_size):
                scaled[offset + i] = obs[offset + i]

        # Return the obs as an ndarray
        vec = np.array(scaled, dtype=np.float32)
        if self.debug > 1:
            print("scale_obs returning vec size = ", vec.shape)
            print(vec)

        return vec
