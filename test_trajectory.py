"""Tests a partial trajectory of the environment model."""

from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper

# Start the ego vehicle in lane 0 (straight through) at a fixed speed
config = {  "debug":            0,
            "init_ego_lane":    0,
            "init_ego_x":       1200.0,
            "init_ego_speed":   33.0
         }

env = SimpleHighwayRampWrapper(config)
env.reset()
traj_reward = 0.0
action = [0.307, 0.03] #scaled
try:
    for step in range(50):
        obs, reward, done, _ = env.step(action)
        traj_reward += reward
        print("///// Step {} complete: scaled accel cmd = {:.4f}, speed = {:.4f}, x = {:.4f}, rew = {:.4f}, done = {}\n"
                .format(step, action[0], obs[env.EGO_SPEED], obs[env.EGO_X], reward, done))
        action[0] *= 0.9
        if done:
            break
    print("///// Test terminated. Accumulated reward = {:.4f}".format(traj_reward))
except Exception as e:
    print("Caught exception: ", e)
