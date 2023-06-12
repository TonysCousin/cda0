from cmath import inf
import sys
import math
import time
from typing import List
import numpy as np
import gymnasium
import ray
import ray.rllib.algorithms.ppo as ppo
#import ray.rllib.algorithms.ddpg as ddpg
#import ray.rllib.algorithms.td3  as td3
#import ray.rllib.algorithms.sac as sac
from ray.tune.logger import pretty_print
import pygame
from pygame.locals import *
from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper
from simple_highway_with_ramp    import Roadway

"""This program runs the selected policy checkpoint for one episode and captures key state variables throughout."""

def main(argv):

    prng = np.random.default_rng()

    # Handle any args
    num_args = len(argv)
    if num_args == 1  or  num_args == 3:
        print("Usage is: {} <checkpoint> [learning_level, starting_lane [, relative_pos]]".format(argv[0]))
        sys.exit(1)

    checkpoint = argv[1]
    learning_level = 0
    relative_pos = 2
    start_lane = int(prng.random()*3)
    if num_args > 2:
        level = int(argv[2])
        if 0 <= level <= 6:
            learning_level = level
        lane = int(argv[3])
        if 0 <= lane <= 2:
            start_lane = lane

        if num_args == 5:
            rp = int(argv[4])
            if 0 <= rp <= 4:
                relative_pos = rp

    ray.init()

    # Set up the environment
    env_config = {  "time_step_size":       0.5,
                    "debug":                0,
                    "difficulty_level":     learning_level,
                    "init_ego_lane":        start_lane,
                    #"neighbor_speed":       29.1,
                    #"neighbor_start_loc":   0.0, #dist downtrack from beginning of lane 1 for n3, m
                    "merge_relative_pos":   relative_pos, #neighbor vehicle that we want ego to be beside when starting in lane 2
                }
    env = SimpleHighwayRampWrapper(env_config)
    #print("///// Environment configured. Params are:")
    #print(pretty_print(cfg.to_dict()))
    env.reset()

    # Algorithm-specific configs - NN structure needs to match the checkpoint being read
    cfg = ppo.PPOConfig()
    cfg.framework("torch").exploration(explore = False)
    model = cfg.to_dict()["model"]
    model["no_final_linear"]                = True
    model["fcnet_hiddens"]                  = [256, 256, 4]
    model["fcnet_activation"]               = "relu"
    model["post_fcnet_activation"]          = "linear"
    cfg.training(model = model)

    """
    cfg = sac.SACConfig()
    cfg.framework("torch").exploration(explore = False)
    cfg_dict = cfg.to_dict()
    policy_config = cfg_dict["policy_model_config"]
    policy_config["fcnet_hiddens"]              = [256, 128]
    policy_config["fcnet_activation"]           = "relu"
    q_config = cfg_dict["q_model_config"]
    q_config["fcnet_hiddens"]                   = [256, 128]
    q_config["fcnet_activation"]                = "relu"
    cfg.training(policy_model_config = policy_config, q_model_config = q_config)
    """

    cfg.environment(env = SimpleHighwayRampWrapper,
                    env_config = env_config
                   )

    # Restore the selected checkpoint file
    # Note that the raw environment class is passed to the algo, but we are only using the algo to run the NN model,
    # not to run the environment, so any environment info we pass to the algo is irrelevant for this program.
    algo = cfg.build()
    algo.restore(checkpoint)
    print("///// Checkpoint {} successfully loaded.".format(checkpoint))
    if learning_level == 4  and  start_lane == 2:
        print("///// using ramp relative position ", relative_pos)

    # Set up the graphic display
    graphics = Graphics(env)

    # Run the agent through a complete episode
    episode_reward = 0
    done = False
    action = [0, 0]
    raw_obs, _ = env.unscaled_reset()
    vehicles = env.get_vehicle_data()
    print("///// Neighbor speeds:  {:.1f}, {:.1f}, {:.1f}".format(vehicles[1].cur_speed, vehicles[2].cur_speed, vehicles[3].cur_speed))
    graphics.update(action, raw_obs, vehicles)
    obs = env.scale_obs(raw_obs)
    step = 0
    time.sleep(2)
    while not done:
        step += 1
        action = algo.compute_single_action(obs, explore = False)
        raw_obs, reward, done, truncated, info = env.step(np.ndarray.tolist(action)) #obs returned is UNSCALED
        episode_reward += reward

        # Display current status of all the vehicles
        graphics.update(action, raw_obs, env.get_vehicle_data())

        # Wait for user to indicate okay to begin animation
        """
        if step == 1:
            print("///// Press Down key to begin...")
            go = False
            while not go:
                keys = pygame.key.get_pressed() #this doesn't work
                if keys[pygame.K_DOWN]:
                    go = True
                    break;
        """

        # Scale the observations to be ready for NN ingest next time step
        obs = env.scale_obs(raw_obs)

        print("///// step {:3d}: scaled action = [{:5.2f} {:5.2f}], lane = {}, speed = {:.2f}, p = {:.3f}, rem = {:4.0f}, r = {:7.4f} {}"
                .format(step, action[0], action[1], raw_obs[0], raw_obs[3], raw_obs[1], raw_obs[2], reward, info["reward_detail"]))

        print("                   Z1    Z2    Z3    Z4    Z5    Z6    Z7    Z8    Z9, neighbor in ego zone = {:3.0f}".format(raw_obs[6]))
        b = 11 #base index of this attribute for Z1 in the obs vector
        s = 5 #size of each zone in the obs vector
        print("      driveable: {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f}"
              .format(raw_obs[b+0*s], raw_obs[b+1*s], raw_obs[b+2*s], raw_obs[b+3*s], raw_obs[b+4*s], raw_obs[b+5*s],
                      raw_obs[b+6*s], raw_obs[b+7*s], raw_obs[b+8*s]))
        b += 1 #base index of this attribute for Z1 in the obs vector
        print("      reachable: {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f}"
              .format(raw_obs[b+0*s], raw_obs[b+1*s], raw_obs[b+2*s], raw_obs[b+3*s], raw_obs[b+4*s], raw_obs[b+5*s],
                      raw_obs[b+6*s], raw_obs[b+7*s], raw_obs[b+8*s]))
        b += 1 #base index of this attribute for Z1 in the obs vector
        print("      occupied:  {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f}"
              .format(raw_obs[b+0*s], raw_obs[b+1*s], raw_obs[b+2*s], raw_obs[b+3*s], raw_obs[b+4*s], raw_obs[b+5*s],
                      raw_obs[b+6*s], raw_obs[b+7*s], raw_obs[b+8*s]))
        b += 1 #base index of this attribute for Z1 in the obs vector
        print("      rel p:     {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f}"
              .format(raw_obs[b+0*s], raw_obs[b+1*s], raw_obs[b+2*s], raw_obs[b+3*s], raw_obs[b+4*s], raw_obs[b+5*s],
                      raw_obs[b+6*s], raw_obs[b+7*s], raw_obs[b+8*s]))
        b += 1 #base index of this attribute for Z1 in the obs vector
        print("      rel speed: {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f}"
              .format(raw_obs[b+0*s], raw_obs[b+1*s], raw_obs[b+2*s], raw_obs[b+3*s], raw_obs[b+4*s], raw_obs[b+5*s],
                      raw_obs[b+6*s], raw_obs[b+7*s], raw_obs[b+8*s]))

        if done:
            print("///// Episode complete: {}. Total reward = {:.2f}".format(info["reason"], episode_reward))
            input("///// Press Enter to close...")
            graphics.close()
            sys.exit()

            # Get user input before closing the window
            for event in pygame.event.get(): #this isn't working
                if event.type == pygame.QUIT:
                    graphics.close()
                    sys.exit()


######################################################################################################
######################################################################################################


class Graphics:

    # set up the colors
    BLACK           = (0, 0, 0)
    WHITE           = (255, 255, 255)
    LANE_EDGE_COLOR = WHITE
    NEIGHBOR_COLOR  = (64, 128, 255)
    EGO_COLOR       = (168, 168, 0) #yellow
    PLOT_AXES_COLOR = (0, 0, 255)

    # Other graphics constants
    LANE_WIDTH = Roadway.WIDTH
    WINDOW_SIZE_X = 1800
    WINDOW_SIZE_Y = 800
    REAL_TIME_RATIO = 5.0   #Factor faster than real time


    def __init__(self,
                 env    : gymnasium.Env
                ):
        """Initializes the graphics and draws the roadway background display."""

        # Save the environment for future reference
        self.env = env

        # set up pygame
        pygame.init()
        self.pgclock = pygame.time.Clock()
        self.display_freq = Graphics.REAL_TIME_RATIO / env.time_step_size

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
        #print("      Graphics init: scale = {}, display center r,s = ({:4d}, {:4d}), roadway center x,y = ({:5.0f}, {:5.0f})"
        #        .format(self.scale, self.display_center_r, self.display_center_s, self.roadway_center_x, self.roadway_center_y))

        # Loop through the lane segments and draw the left and right edge lines of each
        for lane in env.roadway.lanes:
            for seg in lane.segments:
                self._draw_segment(seg[0], seg[1], seg[2], seg[3], Graphics.LANE_WIDTH)

        pygame.display.update()
        #time.sleep(20) #debug only

        # Set up lists of previous screen coords and display colors for each vehicle
        self.prev_veh_r = [0, 0, 0, 0]
        self.prev_veh_s = [0, 0, 0, 0]
        self.veh_colors = [Graphics.EGO_COLOR, Graphics.NEIGHBOR_COLOR, Graphics.NEIGHBOR_COLOR, Graphics.NEIGHBOR_COLOR]

        # Initialize the previous vehicles' locations near the beginning of a lane (doesn't matter which lane for this step)
        for v_idx in range(4):
            self.prev_veh_r[v_idx] = int(self.scale*(self.env.roadway.lanes[0].segments[0][0] - self.roadway_center_x)) + self.display_center_r
            self.prev_veh_s[v_idx] = Graphics.WINDOW_SIZE_Y - \
                                     int(self.scale*(self.env.roadway.lanes[0].segments[0][1] - self.roadway_center_y)) - self.display_center_s
        #TODO: draw rectangles instead of circles, with length = vehicle length & width = 0.5*lane width
        self.veh_radius = int(0.25 * Graphics.LANE_WIDTH * self.scale) #radius of icon in pixels


    def update(self,
               action  : list,      #vector of actions for the ego vehicle for the current time step
               obs     : list,      #vector of observations of the ego vehicle for the current time step
               vehicles: list,      #list of Vehicle objects, with item [0] as the ego vehicle
              ):
        """Paints all updates on the display screen, including the new motion of every vehicle and any data plots."""

        # Loop through each vehicle in the scenario
        for v_idx in range(len(vehicles)):

            # Grab the background under where we want the vehicle to appear & erase the old vehicle
            pygame.draw.circle(self.windowSurface, Graphics.BLACK, (self.prev_veh_r[v_idx], self.prev_veh_s[v_idx]), self.veh_radius, 0)

            # Display the vehicle in its new location.  Note that the obs vector is not scaled at this point.
            new_x, new_y = self._get_vehicle_coords(vehicles, v_idx)
            new_r = int(self.scale*(new_x - self.roadway_center_x)) + self.display_center_r
            new_s = Graphics.WINDOW_SIZE_Y - int(self.scale*(new_y - self.roadway_center_y)) - self.display_center_s
            pygame.draw.circle(self.windowSurface, self.veh_colors[v_idx], (new_r, new_s), self.veh_radius, 0)
            pygame.display.update()
            #print("   // Graphics: moving vehicle {} from r,s = ({:4d}, {:4d}) to ({:4d}, {:4d}) and new x,y = ({:5.0f}, {:5.0f})"
            #        .format(v_idx, self.prev_veh_r[v_idx], self.prev_veh_s[v_idx], new_r, new_s, new_x, new_y))

            # Update the previous location
            self.prev_veh_r[v_idx] = new_r
            self.prev_veh_s[v_idx] = new_s

        # Pause until the next time step
        self.pgclock.tick(self.display_freq)


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

        # Find the scaled lane end-point pixel locations (these is centerline of the lane)
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
        pygame.draw.line(self.windowSurface, Graphics.LANE_EDGE_COLOR, (left_r0, left_s0), (left_r1, left_s1))
        pygame.draw.line(self.windowSurface, Graphics.LANE_EDGE_COLOR, (right_r0, right_s0), (right_r1, right_s1))


    def _get_vehicle_coords(self,
                            vehicles    : List, #list of all Vehicles in the scenario
                            vehicle_id  : int   #ID of the vehicle; 0=ego, 1-3=neighbor vehicles
                           ) -> tuple:
        """Returns the map frame coordinates of the indicated vehicle based on its lane ID and distance downtrack.

            CAUTION: these calcs are hard-coded to the specific roadway geometry in this code,
            it is not a general solution.
        """

        assert 0 <= vehicle_id <= 3, "///// _get_vehicle_coords: invalid vehicle_id = {}".format(vehicle_id)

        road = self.env.roadway
        lane = vehicles[vehicle_id].lane_id
        x = road.param_to_map_frame(vehicles[vehicle_id].p, lane)
        y = None
        if lane < 2:
            y = road.lanes[lane].segments[0][1]
        else:
            ddt = (x - road.lanes[2].start_x)/Roadway.COS_LANE2_ANGLE
            if ddt < road.lanes[2].segments[0][4]: #vehicle is in seg 0
                seg0x0 = road.lanes[2].segments[0][0]
                seg0y0 = road.lanes[2].segments[0][1]
                seg0x1 = road.lanes[2].segments[0][2]
                seg0y1 = road.lanes[2].segments[0][3]

                factor = ddt / road.lanes[2].segments[0][4]
                x = seg0x0 + factor*(seg0x1 - seg0x0)
                y = seg0y0 + factor*(seg0y1 - seg0y0)

            else: #vehicle is in seg 1
                y = road.lanes[2].segments[1][1]

        return x, y


######################################################################################################
######################################################################################################

class Plot:
    """Displays an x-y plot of time series data on the screen."""

    def __init__(self,
                 surface    : pygame.Surface,   #the Pygame surface to draw on
                 corner_x   : int,              #X coordinate of the upper-left corner, screen pixels
                 corner_y   : int,              #Y coordinate of the upper-left corner, screen pixels
                 height     : int,              #height of the plot, pixels
                 width      : int,              #width of the plot, pixels
                 min_y      : float,            #min value of data to be plotted on Y axis
                 max_y      : float,            #max value of data to be plotted on Y axis
                 num_steps  : int       = 150,  #num time steps that will be plotted along X axis
                 color      : tuple     = Graphics.PLOT_AXES_COLOR, #color of the axes
                 title      : str       = None  #Title above the plot
                ):
        """Defines and draws the empty plot on the screen, with axes and title."""

        self.surface = surface
        self.cx = corner_x
        self.cy = corner_y
        self.height = height
        self.width = width
        self.min_y = min_y
        self.max_y = max_y
        self.num_steps = num_steps

        FONT_SIZE = 12

        # Draw the axes
        pygame.draw.line(surface, color, (corner_x, corner_y+height), (corner_x+width, corner_y+height))
        pygame.draw.line(surface, color, (corner_x, corner_y+height), (corner_x, corner_y))

        # Create the plot's text on a separate surface and copy it to the display surface
        if title is not None:
            font = pygame.font.Font('freesans.ttf', FONT_SIZE)
            text = font.render(title, True, color, Graphics.BLACK)
            text_rect = text.get_rect()
            text_rect.center = (corner_x + width//2, corner_y - FONT_SIZE//2)
            surface.blit(text, text_rect)

        pygame.display.update()


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
