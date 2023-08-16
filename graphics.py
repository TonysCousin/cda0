from cmath import inf
import math
import pygame
from pygame.locals import *
import gymnasium
from typing import List
from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper
from simple_highway_with_ramp    import Roadway

"""Provides all the graphics display for the inference program."""

class Graphics:

    # set up the colors
    BLACK           = (0, 0, 0)
    WHITE           = (255, 255, 255)
    LANE_EDGE_COLOR = WHITE
    NEIGHBOR_COLOR  = (64, 128, 255)
    EGO_COLOR       = (168, 168, 0) #yellow
    PLOT_AXES_COLOR = (100,   200, 200)
    DATA_COLOR      = (180, 180, 180)

    # Other graphics constants
    LANE_WIDTH = Roadway.WIDTH
    WINDOW_SIZE_R = 1800        #window width, pixels
    WINDOW_SIZE_S = 800         #window height, pixels
    REAL_TIME_RATIO = 5.0       #Factor faster than real time

    # Geometry of data plots
    PLOT_H          = 150 #height of each plot, pixels
    PLOT_W          = 200 #width of each plot, pixels
    PLOT1_R         = WINDOW_SIZE_R/2 - PLOT_W/2 #corner of plot #1
    PLOT1_S         = WINDOW_SIZE_S/2


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
        self.windowSurface = pygame.display.set_mode((Graphics.WINDOW_SIZE_R, Graphics.WINDOW_SIZE_S), 0, 32)
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
        display_width = Graphics.WINDOW_SIZE_R - 2*buffer
        display_height = Graphics.WINDOW_SIZE_S - 2*buffer
        roadway_width = x_max - x_min
        roadway_height = y_max - y_min
        ar_display = display_width / display_height
        ar_roadway = roadway_width / roadway_height
        self.scale = display_height / roadway_height     #pixels/meter
        if ar_roadway > ar_display:
            self.scale = display_width / roadway_width
        self.roadway_center_x = x_min + 0.5*(x_max - x_min)
        self.roadway_center_y = y_min + 0.5*(y_max - y_min)
        self.display_center_r = Graphics.WINDOW_SIZE_R // 2
        self.display_center_s = Graphics.WINDOW_SIZE_S // 2
        #print("      Graphics init: scale = {}, display center r,s = ({:4d}, {:4d}), roadway center x,y = ({:5.0f}, {:5.0f})"
        #        .format(self.scale, self.display_center_r, self.display_center_s, self.roadway_center_x, self.roadway_center_y))

        # Loop through the lane segments and draw the left and right edge lines of each
        for lane in env.roadway.lanes:
            for seg in lane.segments:
                self._draw_segment(seg[0], seg[1], seg[2], seg[3], Graphics.LANE_WIDTH)

        pygame.display.update()
        #time.sleep(20) #debug only

        # Set up lists of previous screen coords and display colors for each vehicle
        self.prev_veh_r = [0] * (SimpleHighwayRampWrapper.NUM_NEIGHBORS+1)
        self.prev_veh_s = [0] * (SimpleHighwayRampWrapper.NUM_NEIGHBORS+1)
        self.veh_colors = [Graphics.NEIGHBOR_COLOR] * (SimpleHighwayRampWrapper.NUM_NEIGHBORS+1)
        self.veh_colors[0] = Graphics.EGO_COLOR

        # Initialize the previous vehicles' locations near the beginning of a lane (doesn't matter which lane for this step)
        for v_idx in range(len(self.prev_veh_r)):
            self.prev_veh_r[v_idx] = int(self.scale*(self.env.roadway.lanes[0].segments[0][0] - self.roadway_center_x)) + self.display_center_r
            self.prev_veh_s[v_idx] = Graphics.WINDOW_SIZE_S - \
                                     int(self.scale*(self.env.roadway.lanes[0].segments[0][1] - self.roadway_center_y)) - self.display_center_s
        #TODO: draw rectangles instead of circles, with length = vehicle length & width = 0.5*lane width
        self.veh_radius = int(0.25 * Graphics.LANE_WIDTH * self.scale) #radius of icon in pixels

        #
        #..........Add live data plots to the display
        #

        # Plot ego speed
        self.plot_ego_speed = Plot(self.windowSurface, Graphics.PLOT1_R, Graphics.PLOT1_S, Graphics.PLOT_H, Graphics.PLOT_W, 0.0, \
                                   SimpleHighwayRampWrapper.MAX_SPEED, title = "Ego speed")


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

            # Get the vehicle's new location on the surface
            new_x, new_y = self._get_vehicle_coords(vehicles, v_idx)
            new_r = int(self.scale*(new_x - self.roadway_center_x)) + self.display_center_r
            new_s = Graphics.WINDOW_SIZE_S - int(self.scale*(new_y - self.roadway_center_y)) - self.display_center_s

            # If the vehicle is still active display the vehicle in its new location.  Note that the obs vector is not scaled at this point.
            if vehicles[v_idx].active:
                pygame.draw.circle(self.windowSurface, self.veh_colors[v_idx], (new_r, new_s), self.veh_radius, 0)

            # Repaint the surface
            pygame.display.update()
            #print("   // Graphics: moving vehicle {} from r,s = ({:4d}, {:4d}) to ({:4d}, {:4d}) and new x,y = ({:5.0f}, {:5.0f})"
            #        .format(v_idx, self.prev_veh_r[v_idx], self.prev_veh_s[v_idx], new_r, new_s, new_x, new_y))

            # Update the previous location
            self.prev_veh_r[v_idx] = new_r
            self.prev_veh_s[v_idx] = new_s

        # Update data plots
        self.plot_ego_speed.update(vehicles[0].cur_speed)

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
        s0 = Graphics.WINDOW_SIZE_S - (self.scale*(y0 - self.roadway_center_y) + self.display_center_s)
        s1 = Graphics.WINDOW_SIZE_S - (self.scale*(y1 - self.roadway_center_y) + self.display_center_s)

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
                            vehicle_id  : int   #ID of the vehicle; 0=ego, others=neighbor vehicles
                           ) -> tuple:
        """Returns the map frame coordinates of the indicated vehicle based on its lane ID and distance downtrack.

            CAUTION: these calcs are hard-coded to the specific roadway geometry in this code,
            it is not a general solution.
        """

        assert 0 <= vehicle_id < len(vehicles), "///// _get_vehicle_coords: invalid vehicle_id = {}".format(vehicle_id)

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
                 corner_r   : int,              #X coordinate of the upper-left corner, screen pixels
                 corner_s   : int,              #Y coordinate of the upper-left corner, screen pixels
                 height     : int,              #height of the plot, pixels
                 width      : int,              #width of the plot, pixels
                 min_y      : float,            #min value of data to be plotted on Y axis
                 max_y      : float,            #max value of data to be plotted on Y axis
                 max_steps  : int       = 180,  #max num time steps that will be plotted along X axis
                 axis_color : tuple     = Graphics.PLOT_AXES_COLOR, #color of the axes
                 data_color : tuple     = Graphics.DATA_COLOR, #color of the data curve being plotted
                 title      : str       = None  #Title above the plot
                ):
        """Defines and draws the empty plot on the screen, with axes and title."""

        assert max_y > min_y, "///// Plot defined with illegal min_y = {}, max_y = {}".format(min_y, max_y)
        assert max_steps > 0, "///// Plot defined with illegal max_steps = {}".format(max_steps)
        assert corner_r >= 0, "///// Plot defined with illegal corner_r = {}".format(corner_r)
        assert corner_s >= 0, "///// Plot defined with illegal corner_s = {}".format(corner_s)
        assert height > 0,    "///// Plot defined with illegal height = {}".format(height)
        assert width > 0,     "///// Plot defined with illegal width = {}".format(width)

        self.surface = surface
        self.cr = corner_r
        self.cs = corner_s
        self.height = height
        self.width = width
        self.min_y = min_y
        self.max_y = max_y
        self.max_steps = max_steps
        self.data_color = data_color

        FONT_SIZE = 12

        # Determine scale factors for the data
        self.r_scale = self.width / max_steps #pixels per time step
        self.s_scale = self.height / (max_y - min_y) #pixels per unit of data value

        # Initialize drawing coordinates for the data curve (in (r, s) pixel location)
        self.prev_r = None
        self.prev_s = None

        # Draw the axes
        pygame.draw.line(surface, axis_color, (corner_r, corner_s+height), (corner_r+width, corner_s+height))
        pygame.draw.line(surface, axis_color, (corner_r, corner_s+height), (corner_r, corner_s))

        # Create the plot's text on a separate surface and copy it to the display surface
        if title is not None:
            font = pygame.font.Font("/home/starkj/bin/FreeSans.ttf", FONT_SIZE)
            text = font.render(title, True, axis_color, Graphics.BLACK)
            text_rect = text.get_rect()
            text_rect.center = (corner_r + width//2, corner_s - FONT_SIZE//2)
            surface.blit(text, text_rect)

        pygame.display.update()


    def update(self,
               data     : float,    #the real-world data value to be plotted (Y value)
              ):
        """Adds the next sequential data point to the plot."""

        # If there has been no data plotted so far, then set the first point
        if self.prev_r is None:
            self.prev_r = self.cr
            self.prev_s = self.cs + Graphics.PLOT_H - data*self.s_scale

        # Else draw a line from the previous point to the current point
        else:
            new_r = self.prev_r + self.r_scale
            new_s = self.cs + Graphics.PLOT_H - data*self.s_scale
            if new_r <= self.cr + Graphics.PLOT_W:
                pygame.draw.line(self.surface, self.data_color, (self.prev_r, self.prev_s), (new_r, new_s))
                self.prev_r = new_r
                self.prev_s = new_s
                pygame.display.update()
