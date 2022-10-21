import sys
import time
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo
import pygame
from pygame.locals import *
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
    config = ppo.DEFAULT_CONFIG.copy()

    env_config = {  "time_step_size":   0.5,
                    "debug":            0,
                    "init_ego_lane":    start_lane
                }

    model_config = config["model"]
    model_config["fcnet_hiddens"]               = [300, 128, 64]
    #model_config["fcnet_hiddens"]               = [256, 256]

    config["env_config"] = env_config
    config["model"] = model_config
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

    # Set up the graphic display
    graphics = Graphics()

    # Run the agent through a complete episode
    episode_reward = 0
    done = False
    obs = env.reset()
    step = 0
    while not done:
        step += 1
        action = algo.compute_single_action(obs)
        obs, reward, done, info = env.step(np.ndarray.tolist(action))
        episode_reward += reward

        # Display current status
        graphics.update(action, obs)
        print("///// step {:3d}: scaled action = [{:5.2f} {:5.2f}], lane = {}, speed = {:.2f}, dist = {:.3f}, r = {:7.4f} {}"
                .format(step, action[0], action[1], obs[0], obs[2], obs[1], reward, info["reward_detail"]))

        if done:
            graphics.close()
            print("///// Episode complete: {}. Total reward = {:.2f}".format(info["reason"], episode_reward))


class Graphics:

    # set up the colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    # Other graphics constants
    WINDOW_SIZE_X = 1000
    WINDOW_SIZE_Y = 800


    def __init__(self):
        # set up pygame
        pygame.init()

        # set up the window
        self.windowSurface = pygame.display.set_mode((Graphics.WINDOW_SIZE_X, Graphics.WINDOW_SIZE_Y), 0, 32)
        pygame.display.set_caption('cda0')

        # set up fonts
        self.basicFont = pygame.font.SysFont(None, 16)

        # draw the background onto the surface
        self.windowSurface.fill(Graphics.BLACK)

        # Loop through the lanes

            # Loop through each segment of the lane

                # Draw the left & right lines for the segment




    def update(self,
               action  : list,
               obs     : list
              ):
        pass


    def close(self):
        pygame.quit()









if __name__ == "__main__":
   main(sys.argv)
