"""Unit tester for the SimpleHighwayRamp environment model."""

from simple_highway_with_ramp import SimpleHighwayRamp

config = {}

# Case 1 - reset the env
print("\n+++++ Begin case 1")
env = SimpleHighwayRamp(config)
obs = env.reset()
print("Case 1: reset obs = ", obs)

# Case 2 - step with invalid accel command
print("\n+++++ Begin case 2")
action = [-5.4, 0.0]
try:
    env.step(action)
except Exception as e:
    print("Caught exception: ", e)

# Case 3 - step with invalid lane change command
print("\n+++++ Begin case 3")
action = [0.2, 2.0]
try:
    env.step(action)
except Exception as e:
    print("Caught exception: ", e)

# Case 4 - step with positive accel, tiny steering command
print("\n+++++ Begin case 4")
action = [0.83, -0.19]
try:
    env.step(action)
except Exception as e:
    print("Caught exception: ", e)

# Case 5 - another step with positive accel, tiny steering command
print("\n+++++ Begin case 5")
action = [0.98, 0.48]
try:
    env.step(action)
except Exception as e:
    print("Caught exception: ", e)

# Case 6 - v0 makes illegal lane change
print("\n+++++ Begin case 6")
action = [0.0, -0.51]
try:
    env.step(action)
except Exception as e:
    print("Caught exception: ", e)

# Case 7 - collision between v0 and v1 in adjacent lanes
print("\n+++++ Begin case 7")
action = [0.0, -1.0]
env.vehicles[0].dist_downtrack = 481.0
env.vehicles[0].speed = 23.4
env.vehicles[1].dist_downtrack = 802.0
env.vehicles[1].speed = 22.0
try:
    env.step(action)
except Exception as e:
    print("Caught exception: ", e)

# Case 8 - collision between v2 and v1 in same lane
print("\n+++++ Begin case 8")
env.reset()
action = [0.0, 0.0]
env.vehicles[1].dist_downtrack = 55.0
env.vehicles[1].speed = 22.0
env.vehicles[2].dist_downtrack = 49.0
env.vehicles[2].speed = 24.4
try:
    env.step(action)
except Exception as e:
    print("Caught exception: ", e)
