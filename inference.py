from cmath import inf
import sys
import time
from typing import List
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.sac as sac
from ray.tune.logger import pretty_print
from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper
from graphics import Graphics

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
                    #"training":             True,           #TODO: debug only!
                    #"randomize_start_dist": True,           #TODO: debug only
                    #"neighbor_speed":       29.1,
                    #"neighbor_start_loc":   0.0, #dist downtrack from beginning of lane 1 for n3, m
                    "merge_relative_pos":   relative_pos, #neighbor vehicle that we want ego to be beside when starting in lane 2 (level 4 only)
                }
    env = SimpleHighwayRampWrapper(env_config)
    #print("///// Environment configured. Params are:")
    #print(pretty_print(cfg.to_dict()))
    env.reset()

    # Algorithm-specific configs - NN structure needs to match the checkpoint being read
    """
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
    policy_config["fcnet_hiddens"]              = [256, 256]
    policy_config["fcnet_activation"]           = "relu"
    q_config = cfg_dict["q_model_config"]
    q_config["fcnet_hiddens"]                   = [256, 256]
    q_config["fcnet_activation"]                = "relu"
    cfg.training(policy_model_config = policy_config, q_model_config = q_config)

    cfg.environment(env = SimpleHighwayRampWrapper, env_config = env_config)

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
    graphics.update(action, raw_obs, vehicles)
    obs = env.scale_obs(raw_obs)
    step = 0
    time.sleep(4)
    while not done:
        step += 1
        action = algo.compute_single_action(obs, explore = False)

        # Command masking for first few steps to allow feedback obs to populate
        if step < 2:
            action[1] = 0.0

        # Move the environment forward one time step
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

        print("///// step {:3d}: scaled action = [{:5.2f} {:5.2f}], lane = unsp, speed = {:.2f}, p = unsp, rem = {:4.0f}, r = {:7.4f} {}"
                .format(step, action[0], action[1], raw_obs[3], raw_obs[2], reward, info["reward_detail"]))

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

if __name__ == "__main__":
   main(sys.argv)
