import sys
import ray
import ray.rllib.algorithms.ppo as ppo
from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper

"""This program runs the selected policy checkpoint for one episode and captures key state variables throughout."""

def main(args):
    ray.init()

    # Set up the environment
    env_config = {  "time_step_size":   0.5,
                    "debug":            0,
                    "init_ego_lane":    0 #left-most lane, which is just straight
                }

    config = ppo.DEFAULT_CONFIG.copy()
    config["env_configs"] = env_config
    config["framework"] = "torch"
    config["num_gpus"] = 0
    config["num_workers"] = 1
    algo = ppo.PPO(config = config, env = SimpleHighwayRampWrapper)

    # Restore the selected checkpoint file
    algo.restore(args[0])













if __name__ == "__main__":
   main(sys.argv[1:])
