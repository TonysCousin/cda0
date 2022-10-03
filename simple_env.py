"""This code is based upon example from RLlib (https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py)
    and gym documentation (https://www.gymlibrary.dev/content/environment_creation/)
"""

import gym
from gym.spaces import Discrete, Box
import numpy as np
from ray.rllib.env.env_context import EnvContext

class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    metadata = {"render_modes": None}
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, config: EnvContext, seed=None, render_mode=None):
        #print("\n///// SimpleCorridor init: config = ", config)

        #self.seed = seed #Ray 2.0.0 chokes on the seed() method if this is defined (it checks for this attribute also)
        self.render_mode = render_mode

        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2) #0 = back, 1 = forward
        self.observation_space = Box(np.array([0.0]), self.end_pos, shape=(1,), dtype=np.float32)

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
        pass
        #print("///// In SimpleCorridor.seed - incoming seed = ", seed)
        #self.seed = seed
        #super().seed(seed)


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        #self.seed = seed

        # options may be a dict that can specify additional configurations - unique to each particular env
        if options is not None:
            raise ValueError("reset() called with options, but options are not used in this environment.")

        self.cur_pos = 0
        return [self.cur_pos]


    def step(self, action):
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
        # Should provide this method in case resources may need to be released
        pass
