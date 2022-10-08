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
