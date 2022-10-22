from cmath import inf
import sys
import time
import math
import numpy as np
import gym
import ray
import ray.rllib.algorithms.ppo as ppo
import pygame
from pygame.locals import *
from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper
from simple_highway_with_ramp    import SimpleHighwayRamp, Roadway, Lane

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
    model_config["fcnet_hiddens"]               = [300, 128, 64] #needs to be same as in the checkpoint being read!
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
    algo = ppo.PPO(config = config, env = SimpleHighwayRampWrapper) #needs the env class, not the object created above
    algo.restore(checkpoint)
    print("///// Checkpoint {} successfully loaded.".format(checkpoint))

    # Set up the graphic display
    graphics = Graphics(env)

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


######################################################################################################
######################################################################################################


class Graphics:

    # set up the colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    # Other graphics constants
    LANE_WIDTH = 30.0 #m (wider than reality for graphics aesthetics)
    WINDOW_SIZE_X = 1800
    WINDOW_SIZE_Y = 800


    def __init__(self,
                 env    : gym.Env
                ):
        # set up pygame
        pygame.init()

        # set up the window
        self.windowSurface = pygame.display.set_mode((Graphics.WINDOW_SIZE_X, Graphics.WINDOW_SIZE_Y), 0, 32)
        pygame.display.set_caption('cda0')

        # set up fonts
        self.basicFont = pygame.font.SysFont(None, 16)

        # draw the background onto the surface
        self.windowSurface.fill(Graphics.BLACK)

        # Loop through all segments of all lanes and find the extreme coordinates to determine our bounding box
        x_min = inf
        y_min = inf
        x_max = -inf
        y_max = -inf
        for lane in env.roadway.lanes:
            for seg in lane.segments:
                x_min = min(x_min, seg[0], seg[2])
                y_min = min(y_min, seg[1], seg[3])
                x_max = max(x_max, seg[0], seg[2])
                y_max = max(y_max, seg[1], seg[3])

        # Add a buffer all around to ensure we have room to draw the edge lines, which are 1/2 lane width away
        x_min -= 0.5*Graphics.LANE_WIDTH
        y_min -= 0.5*Graphics.LANE_WIDTH
        x_max += 0.5*Graphics.LANE_WIDTH
        y_max += 0.5*Graphics.LANE_WIDTH

        # Define the transform between roadway coords (x, y) and display viewport pixels (r, s).  Note that
        # viewport origin is at upper left, with +s pointing downward.  Leave a few pixels of buffer on all sides
        # of the display so the lines don't bump the edge.
        buffer = 8 #pixels
        display_width = Graphics.WINDOW_SIZE_X - 2*buffer
        display_height = Graphics.WINDOW_SIZE_Y - 2*buffer
        roadway_width = x_max - x_min
        roadway_height = y_max - y_min
        ar_display = display_width / display_height
        ar_roadway = roadway_width / roadway_height
        self.scale = display_height / roadway_height     #pixels/meter
        if ar_roadway > ar_display:
            self.scale = display_width / roadway_width
        self.roadway_center_x = x_min + 0.5*(x_max - x_min)
        self.roadway_center_y = y_min + 0.5*(y_max - y_min)
        self.display_center_r = Graphics.WINDOW_SIZE_X // 2
        self.display_center_s = Graphics.WINDOW_SIZE_Y // 2

        # Loop through the lane segments and draw the left and right edge lines of each
        for lane in env.roadway.lanes:
            for seg in lane.segments:
                self._draw_segment(seg[0], seg[1], seg[2], seg[3], Graphics.LANE_WIDTH)

        pygame.display.update()
        #time.sleep(20) #TODO debug only




    def update(self,
               action  : list,
               obs     : list
              ):
        pass


    def close(self):
        pygame.quit()


    def _draw_segment(self,
                      x0        : float,
                      y0        : float,
                      x1        : float,
                      y1        : float,
                      w         : float
                     ):
        """Draws a single lane segment on the display, which consists of the left and right edge lines.
            ASSUMES that all segments are oriented with headings between 0 and 90 deg for simplicity.
        """

        # Find the scaled end-point pixel locations
        r0 = self.scale*(x0 - self.roadway_center_x) + self.display_center_r
        r1 = self.scale*(x1 - self.roadway_center_x) + self.display_center_r
        s0 = Graphics.WINDOW_SIZE_Y - (self.scale*(y0 - self.roadway_center_y) + self.display_center_s)
        s1 = Graphics.WINDOW_SIZE_Y - (self.scale*(y1 - self.roadway_center_y) + self.display_center_s)

        # Find the scaled width of the lane
        ws = 0.5 * w * self.scale

        angle = math.atan2(y1-y0, x1-x0) #radians in [-pi, pi]
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)

        # Find the screen coords of the left edge line
        left_r0 = r0 - ws*sin_a
        left_r1 = r1 - ws*sin_a
        left_s0 = s0 - ws*cos_a
        left_s1 = s1 - ws*cos_a

        # Find the screen coords of the right edge line
        right_r0 = r0 + ws*sin_a
        right_r1 = r1 + ws*sin_a
        right_s0 = s0 + ws*cos_a
        right_s1 = s1 + ws*cos_a

        # Draw the edge lines
        pygame.draw.line(self.windowSurface, Graphics.WHITE, (left_r0, left_s0), (left_r1, left_s1), 1)
        pygame.draw.line(self.windowSurface, Graphics.WHITE, (right_r0, right_s0), (right_r1, right_s1), 1)


######################################################################################################
######################################################################################################


if __name__ == "__main__":
   main(sys.argv)
