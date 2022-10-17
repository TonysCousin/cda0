import sys
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo
from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper

"""This program runs the selected policy checkpoint for one episode and captures key state variables throughout."""

def main(argv):

    prng = np.random.default_rng()

    # Handle any args
    if len(argv) == 1:
        print("Usage is: {} <checkpoint> [starting lane]".format(argv[0]))
        sys.exit(1)

    checkpoint = argv[1]
    start_lane = int(prng.random()*3)
    if len(argv) > 2:
        lane = int(argv[2])
        if 0 <= lane <= 2:
            start_lane = lane

    ray.init()

    # Set up the environment
    env_config = {  "time_step_size":   0.5,
                    "debug":            0,
                    "init_ego_lane":    start_lane
                }

    config = ppo.DEFAULT_CONFIG.copy()
    config["env_config"] = env_config
    config["framework"] = "torch"
    config["num_gpus"] = 0
    config["num_workers"] = 1
    env = SimpleHighwayRampWrapper(env_config)
    print("///// Environment configured.")

    # Restore the selected checkpoint file
    # Note that the raw environment class is passed to the algo, but we are only using the algo to run the NN model,
    # not to run the environment, so any environment info we pass to the algo is irrelevant for this program.  The
    # algo doesn't recognize the config key "env_configs", so need to remove it here.
    #config.pop("env_configs")
    algo = ppo.PPO(config = config, env = SimpleHighwayRampWrapper)
    algo.restore(checkpoint)
    print("///// Checkpoint {} successfully loaded.".format(checkpoint))

    # Run the agent through a complete episode
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = algo.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        print("///// step: action = {}, lane = {}, speed = {:.2f}, dist = {:.3f}, r = {:.3f}".format(action, obs[0], obs[2], obs[1], reward))
        if done:
            print("///// Episode complete: {}. Total reward = {:.2f}".format(info["reason"], episode_reward))













if __name__ == "__main__":
   main(sys.argv)
